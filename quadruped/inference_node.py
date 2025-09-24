import torch
import numpy as np
from rclpy import init, spin, shutdown, Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
from std_msgs.msg import Float32MultiArray
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import JointState

from .pmtg import ik, trajectory_generator
from .pmtg.trajectory_generator import go2_action_config
from .utils import robot_loader


class InferenceNode(Node):
    def __init__(self):
        super().__init__('inference_node')

        self.declare_parameter('model_path', 'path/to/your/model.pt')
        self.declare_parameter('inference_frequency', 50.0)  # Hz
        self.declare_parameter(
            'urdf_filename', "./urdf/go2_description/urdf/go2_description.urdf")
        self.declare_parameter('package_dir', "./urdf/")

        model_path = self.get_parameter(
            'model_path').get_parameter_value().string_value
        self._load_policy(model_path)
        inference_frequency = self.get_parameter(
            'inference_frequency').get_parameter_value().double_value
        self._inference_period = 1.0 / inference_frequency
        urdf_filename = self.get_parameter(
            'urdf_filename').get_parameter_value().string_value
        package_dir = self.get_parameter(
            'package_dir').get_parameter_value().string_value
        robot = robot_loader.get_pin_robot_wrapper(urdf_filename, package_dir)

        self.observation_subscriber = self.create_subscription(
            Float32MultiArray,
            '/observation',
            self.observation_callback,
            QoSProfile(
                reliability=QoSReliabilityPolicy.BEST_EFFORT,
                history=QoSHistoryPolicy.KEEP_LAST,
                depth=1
            )
        )
        self.joint_state_subscriber = self.create_subscription(
            JointState,
            '/joint_states',
            self.joint_state_callback,
            QoSProfile(
                reliability=QoSReliabilityPolicy.BEST_EFFORT,
                history=QoSHistoryPolicy.KEEP_LAST,
                depth=1
            )
        )
        self.base_pose_subscriber = self.create_subscription(
            PoseStamped,
            '/base_pose',
            self.base_pose_callback,
            QoSProfile(
                reliability=QoSReliabilityPolicy.BEST_EFFORT,
                history=QoSHistoryPolicy.KEEP_LAST,
                depth=1
            )
        )
        self.inference_timer = self.create_timer(
            self._inference_period, self.inference_timer_callback)
        self.action_publisher = self.create_publisher(
            Float32MultiArray,
            '/action',
            QoSProfile(
                reliability=QoSReliabilityPolicy.BEST_EFFORT,
                history=QoSHistoryPolicy.KEEP_LAST,
                depth=1
            )
        )

        self._observation = None
        self._joint_states = None
        self._base_pose = None
        self._action_cfg = go2_action_config()
        self._trajectory_generators = [
            trajectory_generator.HybridFourDimTrajectoryGenerator(
                self._action_cfg.trajectory_generator_params, i)
            for i in range(4)
        ]
        self._phases = torch.zeros(1, 4)
        self._ik_solver = ik.InverseKinematicsSolver(
            robot=robot,
            ee_name_list=['FL_foot', 'FR_foot', 'RL_foot', 'RR_foot'],
            base_name='base',
        )

        self.get_logger().info('InferenceNode initialized.')

    def _load_policy(self, model_path: str):
        """
        Load a pre-trained policy from the specified path.

        Args:
            model_path (str): Path to the saved model.
        """
        self.policy = torch.jit.load(model_path)
        self.policy.eval()
        self.get_logger().info(f'Policy loaded from {model_path}')

    def _compute_action(self, obs: np.ndarray) -> np.ndarray:
        """
        Computes the action from the observation using the loaded policy.

        Args:
            obs (np.ndarray): The observation.

        Returns:
            np.ndarray: The action.
        """
        with torch.no_grad():
            obs = torch.from_numpy(obs).view(1, -1).float()
            action = self.policy(obs).detach().view(-1).numpy()
        return action

    def _compute_joint_targets(self, action: np.ndarray) -> np.ndarray:
        """
        Computes the target joint positions based on the action and current joint states.

        Args:
            action (np.ndarray): The action from the policy.
        Returns:
            np.ndarray: The target joint positions.
        """
        if self._joint_states is None or self._base_pose is None:
            self.get_logger().warning('No joint state or base pose received yet.')
            return None

        tg_args = torch.from_numpy(action[:4]).view(1, -1).float()
        foot_target_positions = []
        for trajectory_generator_idx, trajectory_generator in enumerate(self._trajectory_generators):
            foot_target_position, phase = trajectory_generator.generate(
                tg_args, self._inference_period)
            foot_target_positions.append(foot_target_position)
            self._phases[:, trajectory_generator_idx] = phase

        joint_targets = np.zeros(12)
        for idx, foot in enumerate(['FL_foot', 'FR_foot', 'RL_foot', 'RR_foot']):
            joint_targets[idx * 3: (idx + 1) * 3] = self._ik_solver.solve_ik(
                foot, foot_target_positions[idx], np.array(self._joint_states.position))
            # Add residuals from the policy
            + action[4 + idx * 3: 4 + (idx + 1) * 3] * \
                self._action_cfg.ik_residual_scale

        return joint_targets

    def observation_callback(self, msg: Float32MultiArray):
        """
        Callback function for the observation subscriber.

        Args:
            msg (Float32MultiArray): The incoming observation message.
        """
        self._observation = np.array(msg.data)

    def joint_state_callback(self, msg: JointState):
        """
        Callback function for the joint state subscriber.

        Args:
            msg (JointState): The incoming joint state message.
        """
        self._joint_states = msg

    def base_pose_callback(self, msg: PoseStamped):
        """
        Callback function for the base pose subscriber.

        Args:
            msg (PoseStamped): The incoming base pose message.
        """
        self._base_pose = msg

    def inference_timer_callback(self):
        """
        Timer callback to perform inference and log the action.
        """
        if self._observation is not None:
            action = self._compute_action(self._observation)
            final_action = self._compute_joint_targets(action)
            action_msg = Float32MultiArray()
            action_msg.data = final_action.tolist()
            self.action_publisher.publish(action_msg)
        else:
            self.get_logger().warning('No observation received yet.')


if __name__ == '__main__':
    init()
    node = InferenceNode()
    spin(node)
    shutdown()
