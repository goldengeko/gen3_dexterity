#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import JointState
from moveit_msgs.srv import GetPositionIK
from moveit_msgs.msg import PositionIKRequest, RobotState, MoveItErrorCodes
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from control_msgs.action import FollowJointTrajectory
from tf2_ros import TransformListener, Buffer
from tf2_ros import LookupException, ConnectivityException, ExtrapolationException
from tf_transformations import quaternion_multiply, quaternion_from_euler
from pynput import keyboard

class KeyboardCommander(Node):
    def __init__(self):
        super().__init__('kinova_keyboard_commander')

        # Declare ROS parameters
        self.declare_parameter('linear_increment', 0.02)
        self.declare_parameter('angular_increment', 0.02)
        self.declare_parameter('timeout_duration', 5.0)
        self.linear_increment = self.get_parameter('linear_increment').value
        self.angular_increment = self.get_parameter('angular_increment').value
        self.timeout_duration = self.get_parameter('timeout_duration').value

        # Initialize clients
        self.cli = self.create_client(GetPositionIK, '/compute_ik')
        self.action_client = ActionClient(self, FollowJointTrajectory, '/kinova_joint_trajectory_controller/follow_joint_trajectory')

        # Initialize state variables
        self.filtered_joint_state = None
        self.current_pose = None
        self.is_moving = False
        self.goal_reached = True
        self.last_command = None

        # Subscribers
        self.joint_state_subscriber = self.create_subscription(JointState, '/joint_states', self.joint_state_callback, 10)

        # Wait for services and joint state
        while not self.cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('IK service not available, waiting again...')
        while not self.action_client.wait_for_server(timeout_sec=1.0):
            self.get_logger().info('FollowJointTrajectory action not available, waiting again...')
        while self.filtered_joint_state is None:
            self.get_logger().info('Waiting for filtered joint state...')
            rclpy.spin_once(self)

        # Initialize TF2
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # Timers
        self.tf_timer = self.create_timer(1.0, self.lookup_transform)
        self.timeout_timer = self.create_timer(self.timeout_duration, self.stop_on_timeout)

        # Initialize pynput keyboard listener
        self.listener = keyboard.Listener(on_press=self.on_key_press)
        self.listener.start()
        self.get_logger().info("Keyboard listener started. Use arrow keys, i/j/k/l, s, +/-.")

    def lookup_transform(self):
        try:
            trans = self.tf_buffer.lookup_transform('base_link', 'end_effector_link', rclpy.time.Time())
            translation = trans.transform.translation
            rotation = trans.transform.rotation
            # self.get_logger().info(
            #     f"End Effector Pose:\n"
            #     f"x: {translation.x:.3f}\n"
            #     f"y: {translation.y:.3f}\n"
            #     f"z: {translation.z:.3f}\n"
            #     f"rx: {rotation.x:.3f}\n"
            #     f"ry: {rotation.y:.3f}\n"
            #     f"rz: {rotation.z:.3f}\n"
            #     f"rw: {rotation.w:.3f}"
            # )
            self.current_pose = PoseStamped()
            self.current_pose.header.frame_id = 'base_link'
            self.current_pose.pose.position.x = translation.x
            self.current_pose.pose.position.y = translation.y
            self.current_pose.pose.position.z = translation.z
            self.current_pose.pose.orientation.x = rotation.x
            self.current_pose.pose.orientation.y = rotation.y
            self.current_pose.pose.orientation.z = rotation.z
            self.current_pose.pose.orientation.w = rotation.w
        except (LookupException, ConnectivityException, ExtrapolationException) as e:
            self.get_logger().warn(f"Failed to get end effector pose: {e}")

    def joint_state_callback(self, msg):
        #self.get_logger().info(f"Received joint names: {msg.name}")
        expected_joint_names = ["joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6"]
        filtered_positions = [None] * len(expected_joint_names)
        for i, joint in enumerate(expected_joint_names):
            if joint in msg.name:
                index = msg.name.index(joint)
                filtered_positions[i] = msg.position[index]
        if None in filtered_positions:
            missing_joints = [joint for i, joint in enumerate(expected_joint_names) if filtered_positions[i] is None]
            #self.get_logger().warn(f"Incomplete joint state received. Missing joints: {missing_joints}")
            return
        self.filtered_joint_state = JointState()
        self.filtered_joint_state.name = expected_joint_names
        self.filtered_joint_state.position = filtered_positions
        #self.get_logger().info("Valid joint state received")

    def send_ik_request(self, target_pose):
        if not self.filtered_joint_state:
            self.get_logger().warn("No filtered joint state available")
            return None

        req = GetPositionIK.Request()
        ik_req = PositionIKRequest()
        ik_req.group_name = "manipulator"
        ik_req.ik_link_name = "end_effector_link"
        ik_req.avoid_collisions = True
        ik_req.pose_stamped = target_pose
        ik_req.robot_state = RobotState()
        ik_req.robot_state.joint_state = self.filtered_joint_state
        ik_req.timeout.sec = 5
        req.ik_request = ik_req

        future = self.cli.call_async(req)

        def future_callback(fut):
            try:
                response = fut.result()
                if response.error_code.val == MoveItErrorCodes.SUCCESS:
                    self.get_logger().info("IK solution found")
                    self.send_trajectory(response.solution.joint_state)
                else:
                    error_codes = {
                        MoveItErrorCodes.SUCCESS: "Success",
                        MoveItErrorCodes.NO_IK_SOLUTION: "No IK Solution",
                        MoveItErrorCodes.GOAL_CONSTRAINTS_VIOLATED: "Collision Constraints Violated",
                    }
                    error_msg = error_codes.get(response.error_code.val, 'Unknown Error')
                    self.get_logger().warn(f"IK computation failed: {error_msg}")
            except Exception as e:
                self.get_logger().error(f"Failed to process IK response: {e}")

        future.add_done_callback(future_callback)

    def send_trajectory(self, joint_state):
        if not joint_state:
            self.get_logger().warn("No valid joint state for trajectory")
            return
        expected_joint_names = ["joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6"]
        filtered_positions = [
            joint_state.position[joint_state.name.index(joint)]
            for joint in expected_joint_names if joint in joint_state.name
        ]
        trajectory = JointTrajectory()
        trajectory.joint_names = expected_joint_names
        point = JointTrajectoryPoint()
        point.positions = filtered_positions
        point.velocities = [0.0] * len(filtered_positions)
        point.time_from_start.sec = 3
        trajectory.points.append(point)
        goal = FollowJointTrajectory.Goal()
        goal.trajectory = trajectory
        self.get_logger().info("Sending trajectory to action server")
        self.is_moving = True
        self.goal_reached = False
        goal_handle_future = self.action_client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, goal_handle_future)
        if goal_handle_future.result():
            goal_handle = goal_handle_future.result()
            # result + self.get_logger().info("Trajectory executed successfully")
            self.goal_reached = True
        else:
            self.get_logger().warn("Trajectory execution failed")
        self.is_moving = False

    def move_up_cmd(self):
        if not self.current_pose:
            self.get_logger().warn("Current pose not available")
            return
        self.get_logger().info("Moving up")
        target_pose = self.current_pose
        target_pose.pose.position.z += self.linear_increment
        joint_state = self.send_ik_request(target_pose)
        if joint_state:
            self.send_trajectory(joint_state)
        self.last_command = 'move_up'
        self.reset_timeout_timer()

    def move_down_cmd(self):
        if not self.current_pose:
            self.get_logger().warn("Current pose not available")
            return
        self.get_logger().info("Moving down")
        target_pose = self.current_pose
        target_pose.pose.position.z -= self.linear_increment
        joint_state = self.send_ik_request(target_pose)
        if joint_state:
            self.send_trajectory(joint_state)
        self.last_command = 'move_down'
        self.reset_timeout_timer()

    def move_left_cmd(self):
        if not self.current_pose:
            self.get_logger().warn("Current pose not available")
            return
        self.get_logger().info("Moving left")
        target_pose = self.current_pose
        target_pose.pose.position.y += self.linear_increment
        joint_state = self.send_ik_request(target_pose)
        if joint_state:
            self.send_trajectory(joint_state)
        self.last_command = 'move_left'
        self.reset_timeout_timer()

    def move_right_cmd(self):
        if not self.current_pose:
            self.get_logger().warn("Current pose not available")
            return
        self.get_logger().info("Moving right")
        target_pose = self.current_pose
        target_pose.pose.position.y -= self.linear_increment
        joint_state = self.send_ik_request(target_pose)
        if joint_state:
            self.send_trajectory(joint_state)
        self.last_command = 'move_right'
        self.reset_timeout_timer()

    def look_up_cmd(self):
        if not self.current_pose:
            self.get_logger().warn("Current pose not available")
            return
        self.get_logger().info("Looking up")
        target_pose = self.current_pose
        current_quat = [target_pose.pose.orientation.x, target_pose.pose.orientation.y,
                        target_pose.pose.orientation.z, target_pose.pose.orientation.w]
        rotation_quat = quaternion_from_euler(0, self.angular_increment, 0)
        new_quat = quaternion_multiply(current_quat, rotation_quat)
        target_pose.pose.orientation.x, target_pose.pose.orientation.y, \
        target_pose.pose.orientation.z, target_pose.pose.orientation.w = new_quat
        joint_state = self.send_ik_request(target_pose)
        if joint_state:
            self.send_trajectory(joint_state)
        self.last_command = 'look_up'
        self.reset_timeout_timer()

    def look_down_cmd(self):
        if not self.current_pose:
            self.get_logger().warn("Current pose not available")
            return
        self.get_logger().info("Looking down")
        target_pose = self.current_pose
        current_quat = [target_pose.pose.orientation.x, target_pose.pose.orientation.y,
                        target_pose.pose.orientation.z, target_pose.pose.orientation.w]
        rotation_quat = quaternion_from_euler(0, -self.angular_increment, 0)
        new_quat = quaternion_multiply(current_quat, rotation_quat)
        target_pose.pose.orientation.x, target_pose.pose.orientation.y, \
        target_pose.pose.orientation.z, target_pose.pose.orientation.w = new_quat
        joint_state = self.send_ik_request(target_pose)
        if joint_state:
            self.send_trajectory(joint_state)
        self.last_command = 'look_down'
        self.reset_timeout_timer()

    def look_left_cmd(self):
        if not self.current_pose:
            self.get_logger().warn("Current pose not available")
            return
        self.get_logger().info("Looking left")
        target_pose = self.current_pose
        current_quat = [target_pose.pose.orientation.x, target_pose.pose.orientation.y,
                        target_pose.pose.orientation.z, target_pose.pose.orientation.w]
        rotation_quat = quaternion_from_euler(0, 0, self.angular_increment)
        new_quat = quaternion_multiply(current_quat, rotation_quat)
        target_pose.pose.orientation.x, target_pose.pose.orientation.y, \
        target_pose.pose.orientation.z, target_pose.pose.orientation.w = new_quat
        joint_state = self.send_ik_request(target_pose)
        if joint_state:
            self.send_trajectory(joint_state)
        self.last_command = 'look_left'
        self.reset_timeout_timer()

    def look_right_cmd(self):
        if not self.current_pose:
            self.get_logger().warn("Current pose not available")
            return
        self.get_logger().info("Looking right")
        target_pose = self.current_pose
        current_quat = [target_pose.pose.orientation.x, target_pose.pose.orientation.y,
                        target_pose.pose.orientation.z, target_pose.pose.orientation.w]
        rotation_quat = quaternion_from_euler(0, 0, -self.angular_increment)
        new_quat = quaternion_multiply(current_quat, rotation_quat)
        target_pose.pose.orientation.x, target_pose.pose.orientation.y, \
        target_pose.pose.orientation.z, target_pose.pose.orientation.w = new_quat
        joint_state = self.send_ik_request(target_pose)
        if joint_state:
            self.send_trajectory(joint_state)
        self.last_command = 'look_right'
        self.reset_timeout_timer()

    def stop_movement(self):
        if not self.current_pose or not self.filtered_joint_state:
            self.get_logger().warn("Cannot stop: pose or joint state unavailable")
            return
        self.get_logger().info("Stopping movement")
        joint_state = self.send_ik_request(self.current_pose)
        if joint_state:
            self.send_trajectory(joint_state)
        self.is_moving = False
        self.last_command = 'stop'
        self.reset_timeout_timer()

    def adjust_speed(self, delta):
        self.linear_increment = max(0.01, self.linear_increment + delta)
        self.angular_increment = max(0.01, self.angular_increment + delta)
        self.get_logger().info(f"New linear increment: {self.linear_increment}, angular increment: {self.angular_increment}")
        self.reset_timeout_timer()

    def stop_on_timeout(self):
        if not self.goal_reached:
            self.get_logger().info("Timeout reached. Stopping movement")
            self.stop_movement()

    def reset_timeout_timer(self):
        self.timeout_timer.cancel()
        self.timeout_timer = self.create_timer(self.timeout_duration, self.stop_on_timeout)

    def on_key_press(self, key):
        if not self.goal_reached:
            return  # Wait for current trajectory to complete
        try:
            # Handle special keys
            if hasattr(key, 'name'):
                key_str = key.name
            else:
                key_str = str(key.char)
            # Map keys to commands
            key_map = {
                'up': self.move_up_cmd,
                'down': self.move_down_cmd,
                'left': self.move_left_cmd,
                'right': self.move_right_cmd,
                'i': self.look_up_cmd,
                'k': self.look_down_cmd,
                'j': self.look_left_cmd,
                'l': self.look_right_cmd,
                's': self.stop_movement,
                'plus': lambda: self.adjust_speed(0.05),
                'minus': lambda: self.adjust_speed(-0.05),
            }
            if key_str in key_map:
                key_map[key_str]()
                self.get_logger().info(f"Processed key: {key_str}")
            else:
                self.get_logger().info(f"Ignored key: {key_str}")
        except AttributeError:
            pass

    def destroy_node(self):
        self.listener.stop()
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    commander = KeyboardCommander()
    try:
        rclpy.spin(commander)
    except KeyboardInterrupt:
        commander.get_logger().info('Keyboard interrupt, shutting down...')
    except Exception as e:
        commander.get_logger().error(f"Unexpected error: {e}")
    finally:
        commander.destroy_node()
        rclpy.shutdown()
        commander.get_logger().info('Node destroyed, shutting down...')

if __name__ == '__main__':
    main()
