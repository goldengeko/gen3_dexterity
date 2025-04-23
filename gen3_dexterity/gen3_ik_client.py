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

class IKClientNode(Node):
    def __init__(self):
        super().__init__('ik_client_node')
        self.cli = self.create_client(GetPositionIK, '/compute_ik')
        self.action_client = ActionClient(self, FollowJointTrajectory, '/kinova_joint_trajectory_controller/follow_joint_trajectory')

        # Initialize a variable to store the filtered current joint state
        self.filtered_joint_state = None

        # Subscribe to the /joint_states topic
        self.create_subscription(JointState, '/joint_states', self.joint_state_callback, 10)

        while not self.cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Waiting for /compute_ik service...')

        while not self.action_client.wait_for_server(timeout_sec=1.0):
            self.get_logger().info('Waiting for FollowJointTrajectory action server...')

        # Wait until the filtered joint state is received
        while self.filtered_joint_state is None:
            self.get_logger().info('Waiting for filtered joint state...')
            rclpy.spin_once(self)

        self.send_ik_request()

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # Timer to periodically look up the end effector pose
        self.timer = self.create_timer(1.0, self.lookup_transform)

    def lookup_transform(self):
        try:
            # Look up transform from base_link to end_effector
            trans = self.tf_buffer.lookup_transform('base_link', 'end_effector_link', rclpy.time.Time())
            translation = trans.transform.translation
            rotation = trans.transform.rotation
            self.get_logger().info(
                f"End Effector Pose:\n"
                f"x: {translation.x:.3f}\n"
                f"y: {translation.y:.3f}\n"
                f"z: {translation.z:.3f}\n"
                f"rx: {rotation.x:.3f}\n"
                f"ry: {rotation.y:.3f}\n"
                f"rz: {rotation.z:.3f}\n"
                f"rw: {rotation.w:.3f}"
            )
        except (LookupException, ConnectivityException, ExtrapolationException) as e:
            self.get_logger().warn(f"Failed to get end effector pose: {e}")

    def joint_state_callback(self, msg):
        # Define the expected joint names for the manipulator
        expected_joint_names = [
            "joint_1",
            "joint_2",
            "joint_3",
            "joint_4",
            "joint_5",
            "joint_6",
        ]

        # Initialize positions with None for missing joints
        filtered_positions = [None] * len(expected_joint_names)

        # Map the incoming joint names to their positions
        for i, joint in enumerate(expected_joint_names):
            if joint in msg.name:
                index = msg.name.index(joint)
                filtered_positions[i] = msg.position[index]

        # Check if all positions are populated
        if None in filtered_positions:
            self.get_logger().warn("Incomplete joint state received. Waiting for all joints.")
            return

        # Create a filtered joint state
        self.filtered_joint_state = JointState()
        self.filtered_joint_state.name = expected_joint_names
        self.filtered_joint_state.position = filtered_positions

    def send_ik_request(self):
        req = GetPositionIK.Request()
        ik_req = PositionIKRequest()

        # Specify the group name and link name for the arm
        ik_req.group_name = 'manipulator'
        ik_req.ik_link_name = 'end_effector_link'

        # Enable collision checking
        ik_req.avoid_collisions = True

        # Define the target pose
        pose = PoseStamped()
        pose.header.frame_id = 'base_link'
        pose.pose.position.x = 0.8
        pose.pose.position.y = 0.0
        pose.pose.position.z = 0.25
        pose.pose.orientation.x = 0.5
        pose.pose.orientation.y = 0.5
        pose.pose.orientation.z = 0.5  # Example quaternion for 90 degrees rotation around Z-axis
        pose.pose.orientation.w = 0.5 # Example quaternion for 90 degrees rotation around Z-axis
        ik_req.pose_stamped = pose

        # Use the dynamically filtered joint state
        robot_state = RobotState()
        robot_state.joint_state = self.filtered_joint_state
        ik_req.robot_state = robot_state

        # Set timeout for IK computation
        ik_req.timeout.sec = 5
        ik_req.timeout.nanosec = 0

        req.ik_request = ik_req

        # Log the target pose and joint state
        self.get_logger().info(
            f"Target Pose:\n"
            f"x: {pose.pose.position.x:.3f}\n"
            f"y: {pose.pose.position.y:.3f}\n"
            f"z: {pose.pose.position.z:.3f}\n"
            f"rx: {pose.pose.orientation.x:.3f}\n"
            f"ry: {pose.pose.orientation.y:.3f}\n"
            f"rz: {pose.pose.orientation.z:.3f}\n"
            f"rw: {pose.pose.orientation.w:.3f}"
        )
        self.get_logger().debug(
            f"Filtered Joint State:\n"
            f"Names: {self.filtered_joint_state.name}\n"
            f"Positions: {self.filtered_joint_state.position}"
        )

        # Call the IK service
        future = self.cli.call_async(req)
        rclpy.spin_until_future_complete(self, future)

        if future.result() is not None:
            result = future.result()
            if result.error_code.val == 1:
                joints = result.solution.joint_state

                # Filter joint names and positions for joints 1 to 6
                relevant_joint_names = ["joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6"]
                filtered_joint_positions = [
                    joints.position[joints.name.index(joint)]
                    for joint in relevant_joint_names
                    if joint in joints.name
                ]

                # Log each joint name and position on a separate line
                self.get_logger().info("IK Solution Found:")
                for name, position in zip(relevant_joint_names, filtered_joint_positions):
                    self.get_logger().info(f"{name}: {position:.3f}")

                self.send_trajectory(joints)
            else:
                error_codes = {
                    MoveItErrorCodes.SUCCESS: "Success",
                    MoveItErrorCodes.NO_IK_SOLUTION: "No IK Solution",
                    MoveItErrorCodes.GOAL_CONSTRAINTS_VIOLATED: "Collision Constraints Violated",
                    # Add other error codes as needed
                }
                self.get_logger().warn(
                    f"IK failed with error code: {result.error_code.val} "
                    f"({error_codes.get(result.error_code.val, 'Unknown Error')})"
                )
        else:
            self.get_logger().error("Service call failed")

    def send_trajectory(self, joint_state):
        # Define the expected joint names for the controller
        expected_joint_names = [
            "joint_1",
            "joint_2",
            "joint_3",
            "joint_4",
            "joint_5",
            "joint_6",
        ]

        # Filter the joint names and positions to match the expected joints
        filtered_positions = [
            joint_state.position[joint_state.name.index(joint)]
            for joint in expected_joint_names
            if joint in joint_state.name
        ]

        # Create a JointTrajectory message
        trajectory = JointTrajectory()
        trajectory.joint_names = expected_joint_names

        # Create a single trajectory point
        point = JointTrajectoryPoint()
        point.positions = filtered_positions
        point.velocities = [0.0] * len(filtered_positions)  # Set velocities to 0
        point.time_from_start.sec = 3  # Set a duration for the motion

        trajectory.points.append(point)

        # Create a goal message for the action
        goal_msg = FollowJointTrajectory.Goal()
        goal_msg.trajectory = trajectory

        # Send the trajectory using the action client
        self.get_logger().info("Sending trajectory!")
        self.action_client.send_goal_async(goal_msg)


def main(args=None):
    rclpy.init(args=args)
    node = IKClientNode()
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
