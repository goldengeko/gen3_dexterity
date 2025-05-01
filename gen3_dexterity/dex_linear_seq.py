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
from copy import deepcopy
from enum import Enum
from typing import Callable, List, Optional
import time

class MovementType(Enum):
    TRANSLATION = "translation"
    ROTATION = "rotation"

class LinearCommander(Node):
    def __init__(self):
        super().__init__('linear_commander')
        self._initialize_parameters()
        self._initialize_clients()
        self._initialize_state()
        self._initialize_subscribers_and_timers()
        self._wait_for_services_and_state()

    def _initialize_parameters(self):
        """Initialize node parameters."""
        self.declare_parameter('horizontal_increment', 0.35)
        self.declare_parameter('vertical_increment', 0.15)
        self.declare_parameter('translational_increment', 0.05)
        self.declare_parameter('angular_increment', 1.0)
        self.declare_parameter('timeout_duration', 5.0)
        self.declare_parameter('joint_names', ["joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6"])
        self.horizontal_increment = self.get_parameter('horizontal_increment').value
        self.vertical_increment = self.get_parameter('vertical_increment').value
        self.trans_increment = self.get_parameter('translational_increment').value
        self.angular_increment = self.get_parameter('angular_increment').value
        self.timeout_duration = self.get_parameter('timeout_duration').value
        self.expected_joint_names = self.get_parameter('joint_names').value

    def _initialize_clients(self):
        self.cli = self.create_client(GetPositionIK, '/compute_ik')
        self.action_client = ActionClient(self, FollowJointTrajectory, '/kinova_joint_trajectory_controller/follow_joint_trajectory')

    def _initialize_state(self):
        self.filtered_joint_state = None
        self.current_pose = None
        self.is_moving = False
        self.goal_reached = True
        self.last_command = None
        self.goal_handle = None

        self.home_pose = PoseStamped()
        self.home_pose.header.frame_id = 'base_link'
        self.home_pose.pose.position.x = 0.73
        self.home_pose.pose.position.y = 0.0
        self.home_pose.pose.position.z = 0.39
        self.home_pose.pose.orientation.x = 0.51
        self.home_pose.pose.orientation.y = 0.51
        self.home_pose.pose.orientation.z = 0.5
        self.home_pose.pose.orientation.w = 0.49

    def _initialize_subscribers_and_timers(self):
        self.joint_state_subscriber = self.create_subscription(JointState, '/joint_states', self.joint_state_callback, 10)
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.tf_timer = self.create_timer(0.1, self.lookup_transform)
        self.timeout_timer = self.create_timer(self.timeout_duration, self.stop_on_timeout)

    def _wait_for_services_and_state(self):
        """Wait for services and initial state with timeout."""
        def wait_with_timeout(condition, log_msg, timeout_sec=10):
            start_time = self.get_clock().now()
            while not condition():
                self.get_logger().info(log_msg)
                rclpy.spin_once(self)
                if (self.get_clock().now() - start_time).nanoseconds > timeout_sec * 1e9:
                    self.get_logger().warn(f"Timeout waiting for {log_msg.lower()}")
                    return False
            return True

        wait_with_timeout(lambda: self.cli.wait_for_service(timeout_sec=1.0), "IK service")
        wait_with_timeout(lambda: self.action_client.wait_for_server(timeout_sec=1.0), "FollowJointTrajectory action")
        wait_with_timeout(lambda: self.filtered_joint_state is not None, "filtered joint state")

        if not wait_with_timeout(lambda: self.current_pose is not None, "initial pose"):
            self.get_logger().warn("Initial pose not available, using default pose")
            self.current_pose = deepcopy(self.home_pose)

    def lookup_transform(self):
        try:
            trans = self.tf_buffer.lookup_transform('base_link', 'end_effector_link', rclpy.time.Time(), timeout=rclpy.duration.Duration(seconds=2.0))
            self.current_pose = PoseStamped()
            self.current_pose.header.frame_id = 'base_link'
            self.current_pose.pose.position.x = trans.transform.translation.x
            self.current_pose.pose.position.y = trans.transform.translation.y
            self.current_pose.pose.position.z = trans.transform.translation.z
            self.current_pose.pose.orientation.x = trans.transform.rotation.x
            self.current_pose.pose.orientation.y = trans.transform.rotation.y
            self.current_pose.pose.orientation.z = trans.transform.rotation.z
            self.current_pose.pose.orientation.w = trans.transform.rotation.w
            self.get_logger().debug("Successfully updated current_pose")
        except (LookupException, ConnectivityException, ExtrapolationException) as e:
            self.get_logger().warn(f"Failed to get end effector pose: {e}")

    def joint_state_callback(self, msg: JointState):
        """Filter joint state for expected joints."""
        filtered_positions = [None] * len(self.expected_joint_names)
        for i, joint in enumerate(self.expected_joint_names):
            if joint in msg.name:
                filtered_positions[i] = msg.position[msg.name.index(joint)]
        if None in filtered_positions:
            self.get_logger().debug(f"Incomplete joint state, missing: {[joint for i, joint in enumerate(self.expected_joint_names) if filtered_positions[i] is None]}")
            return
        self.filtered_joint_state = JointState()
        self.filtered_joint_state.name = self.expected_joint_names
        self.filtered_joint_state.position = filtered_positions
        self.get_logger().debug("Valid joint state received")

    def execute_movement(self, movement_type: MovementType, axis: str, increment: float, command_name: str):
        if not self.current_pose or not self.filtered_joint_state:
            self.get_logger().warn(f"Cannot execute {command_name}: pose or joint state unavailable")
            self.goal_reached = True
            return

        target_pose = deepcopy(self.current_pose)
        if movement_type == MovementType.TRANSLATION:
            setattr(target_pose.pose.position, axis, getattr(target_pose.pose.position, axis) + increment)
        elif movement_type == MovementType.ROTATION:
            current_quat = [target_pose.pose.orientation.x, target_pose.pose.orientation.y,
                            target_pose.pose.orientation.z, target_pose.pose.orientation.w]
            axis_map = {'x': (increment, 0, 0), 'y': (0, increment, 0), 'z': (0, 0, increment)}
            rotation_quat = quaternion_from_euler(*axis_map[axis])
            new_quat = quaternion_multiply(current_quat, rotation_quat)
            target_pose.pose.orientation.x, target_pose.pose.orientation.y, \
            target_pose.pose.orientation.z, target_pose.pose.orientation.w = new_quat

        self.get_logger().debug(f"Executing {command_name}")
        self.send_ik_request(target_pose, self.send_trajectory)
        self.last_command = command_name
        self.reset_timeout_timer()

    def home_cmd(self):
        """Move to home position."""
        if not self.current_pose or not self.filtered_joint_state:
            self.get_logger().warn("Cannot move to home: pose or joint state unavailable")
            self.goal_reached = True
            return
        self.get_logger().debug("Moving to home position")
        self.send_ik_request(deepcopy(self.home_pose), self.send_trajectory)
        self.last_command = 'home'
        self.reset_timeout_timer()

    def send_ik_request(self, target_pose: PoseStamped, callback: Callable):
        if not self.filtered_joint_state:
            self.get_logger().warn("No filtered joint state available")
            callback(None)
            return
        req = GetPositionIK.Request()
        ik_req = PositionIKRequest()
        ik_req.group_name = "manipulator"
        ik_req.ik_link_name = "end_effector_link"
        ik_req.avoid_collisions = True
        target_pose.header.stamp = self.get_clock().now().to_msg()
        ik_req.pose_stamped = target_pose
        ik_req.robot_state = RobotState()
        ik_req.robot_state.joint_state = self.filtered_joint_state
        ik_req.timeout.sec = 5
        req.ik_request = ik_req
        self.get_logger().debug("Sending IK request")
        future = self.cli.call_async(req)
        future.add_done_callback(lambda fut: self.ik_callback(fut, callback))

    def ik_callback(self, future, callback: Callable):
        try:
            response = future.result()
            self.get_logger().debug(f"IK response: {response.error_code.val}")
            if response.error_code.val == MoveItErrorCodes.SUCCESS:
                self.get_logger().info("IK solution found")
                callback(response.solution.joint_state)
            else:
                error_msg = {
                    MoveItErrorCodes.SUCCESS: "Success",
                    MoveItErrorCodes.NO_IK_SOLUTION: "No IK Solution",
                    MoveItErrorCodes.GOAL_CONSTRAINTS_VIOLATED: "Collision Constraints Violated",
                }.get(response.error_code.val, 'Unknown Error')
                self.get_logger().warn(f"IK computation failed: {error_msg}")
                callback(None)
        except Exception as e:
            self.get_logger().error(f"Failed to process IK response: {e}")
            callback(None)

    def send_trajectory(self, joint_state: Optional[JointState]):
        if not joint_state:
            self.get_logger().warn("No valid joint state for trajectory")
            self.goal_reached = True
            return
        filtered_positions = [
            joint_state.position[joint_state.name.index(joint)]
            for joint in self.expected_joint_names if joint in joint_state.name
        ]
        trajectory = JointTrajectory()
        trajectory.joint_names = self.expected_joint_names
        point = JointTrajectoryPoint()
        point.positions = filtered_positions
        point.velocities = [0.0] * len(filtered_positions)
        point.time_from_start.sec = 3
        trajectory.points.append(point)
        goal = FollowJointTrajectory.Goal()
        goal.trajectory = trajectory
        self.get_logger().debug("Sending trajectory")
        self.is_moving = True
        self.goal_reached = False
        goal_handle_future = self.action_client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, goal_handle_future, timeout_sec=5.0)
        if not goal_handle_future.result():
            self.get_logger().warn("Failed to send trajectory goal")
            self.is_moving = False
            self.goal_reached = True
            return
        self.goal_handle = goal_handle_future.result()
        result_future = self.goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future, timeout_sec=5.0)
        if result_future.result() and result_future.result().result.error_code == 0:
            self.get_logger().info("Trajectory executed successfully")
            self.goal_reached = True
        else:
            self.get_logger().warn(f"Trajectory execution failed: {result_future.result().result.error_code if result_future.result() else 'Timeout'}")
            self.goal_reached = True
        self.is_moving = False
        self.goal_handle = None

    def stop_on_timeout(self):
        if not self.goal_reached:
            self.get_logger().debug("Timeout reached. Stopping movement")
            self.execute_movement(MovementType.TRANSLATION, 'x', 0.0, 'stop')
            if self.goal_handle:
                cancel_future = self.goal_handle.cancel_goal_async()
                rclpy.spin_until_future_complete(self, cancel_future)
                self.get_logger().info("Active trajectory goal canceled")

    def reset_timeout_timer(self):
        """Reset the timeout timer."""
        self.timeout_timer.cancel()
        self.timeout_timer = self.create_timer(self.timeout_duration, self.stop_on_timeout)

    def execute_sequence(self, sequence_name: str, commands: List[Callable]):
        self.get_logger().info(f"Starting {sequence_name} sequence")
        for cmd in commands:
            cmd_name = cmd.__name__ if hasattr(cmd, '__name__') else str(cmd)
            self.get_logger().info(f"Executing command: {cmd_name}")
            self.goal_reached = False
            cmd()
            start_time = self.get_clock().now()
            while not self.goal_reached:
                rclpy.spin_once(self, timeout_sec=0.1)
                if (self.get_clock().now() - start_time).nanoseconds > 20 * 1e9:
                    self.get_logger().warn(f"Timeout waiting for command {cmd_name}, proceeding to next")
                    self.goal_reached = True
                    break
            self.get_logger().info(f"Command {cmd_name} completed")
            self.get_logger().info("Pausing for 1 second")
            time.sleep(1.0)
    
    def execute_diagonal_movement(self, y_increment: float, z_increment: float, yaw_increment: float, pitch_increment: float, command_name: str):
        """Execute a diagonal movement in the y-z plane with yaw and pitch rotations."""
        if not self.current_pose or not self.filtered_joint_state:
            self.get_logger().warn(f"Cannot execute {command_name}: pose or joint state unavailable")
            self.goal_reached = True
            return

        target_pose = deepcopy(self.current_pose)
        target_pose.pose.position.y += y_increment
        target_pose.pose.position.z += z_increment

        current_quat = [
            target_pose.pose.orientation.x,
            target_pose.pose.orientation.y,
            target_pose.pose.orientation.z,
            target_pose.pose.orientation.w
        ]
        # Apply yaw (y-axis) then pitch (x-axis) rotation
        yaw_quat = quaternion_from_euler(0, yaw_increment, 0)
        pitch_quat = quaternion_from_euler(pitch_increment, 0, 0)
        # Combine: yaw first, then pitch
        intermediate_quat = quaternion_multiply(current_quat, yaw_quat)
        new_quat = quaternion_multiply(intermediate_quat, pitch_quat)
        target_pose.pose.orientation.x, target_pose.pose.orientation.y, \
        target_pose.pose.orientation.z, target_pose.pose.orientation.w = new_quat

        self.get_logger().debug(f"Executing {command_name}")
        self.send_ik_request(target_pose, self.send_trajectory)
        self.last_command = command_name
        self.reset_timeout_timer()

    def linear_sequence(self):
        commands = [
            lambda: self.execute_movement(MovementType.TRANSLATION, 'y', -self.horizontal_increment, 'move_right'),
            lambda: self.execute_movement(MovementType.TRANSLATION, 'x', self.trans_increment, 'move_forward'),
            lambda: self.execute_movement(MovementType.ROTATION, 'y', self.angular_increment, 'look_left'),
            self.home_cmd,
            lambda: self.execute_movement(MovementType.TRANSLATION, 'y', self.horizontal_increment, 'move_left'),
            lambda: self.execute_movement(MovementType.TRANSLATION, 'x', self.trans_increment, 'move_forward'),
            lambda: self.execute_movement(MovementType.ROTATION, 'y', -self.angular_increment, 'look_right'),
            self.home_cmd
        ]
        self.execute_sequence("linear", commands)

    def omni_sequence(self):
        """Execute X pattern sequence (diagonals of a square in y-z plane)."""
        commands = [
        # Top-left (y=+0.15, z=+0.15): Face up-right (positive yaw and pitch)
        lambda: self.execute_diagonal_movement(
            y_increment=-self.horizontal_increment,
            z_increment=self.vertical_increment,
            yaw_increment=self.angular_increment * 0.5,  # Approx 45° yaw
            pitch_increment=self.angular_increment * 0.5,  # Approx 45° pitch
            command_name='move_top_left'
        ),
        lambda: self.execute_movement(MovementType.ROTATION, 'x', 0.0, 'no_op_rotation'),  # Placeholder for triplet structure
        self.home_cmd,
        # Bottom-right (y=-0.15, z=-0.15): Face down-left (negative yaw and pitch)
        lambda: self.execute_diagonal_movement(
            y_increment=self.horizontal_increment,
            z_increment=-self.vertical_increment,
            yaw_increment=-self.angular_increment * 0.5,
            pitch_increment=-self.angular_increment * 0.5,
            command_name='move_bottom_right'
        ),
        lambda: self.execute_movement(MovementType.ROTATION, 'x', 0.0, 'no_op_rotation'),
        self.home_cmd,
        # Top-right (y=-0.15, z=+0.15): Face up-left (negative yaw, positive pitch)
        lambda: self.execute_diagonal_movement(
            y_increment=self.horizontal_increment,
            z_increment=self.vertical_increment,
            yaw_increment=-self.angular_increment * 0.5,
            pitch_increment=self.angular_increment * 0.5,
            command_name='move_top_right'
        ),
        lambda: self.execute_movement(MovementType.ROTATION, 'x', 0.0, 'no_op_rotation'),
        self.home_cmd,
        # Bottom-left (y=+0.15, z=-0.15): Face down-right (positive yaw, negative pitch)
        lambda: self.execute_diagonal_movement(
            y_increment=-self.horizontal_increment,
            z_increment=-self.vertical_increment,
            yaw_increment=self.angular_increment * 0.5,
            pitch_increment=-self.angular_increment * 0.5,
            command_name='move_bottom_left'
        ),
        lambda: self.execute_movement(MovementType.ROTATION, 'x', 0.0, 'no_op_rotation'),
        self.home_cmd
    ]
        self.execute_sequence("x_pattern", commands)

    def destroy_node(self):
        """Clean up node resources."""
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    commander = LinearCommander()
    try:
        commander.omni_sequence()
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
