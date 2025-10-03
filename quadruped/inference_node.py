import torch
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
from std_msgs.msg import Float64MultiArray
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import JointState
from ament_index_python.packages import get_package_share_directory

from .pmtg import ik, trajectory_generator
from .pmtg.trajectory_generator import go2_action_config
from .utils import robot_loader, message_processor

import debugpy

debugpy.listen(5678)
print("Waiting for debugger attach...")
debugpy.wait_for_client()
print("Debugger attached")


class InferenceNode(Node):
    def __init__(self):
        super().__init__('inference_node')

        self.declare_parameter('model_path', get_package_share_directory(
            "quadruped") + "/policies/policy.pt")
        self.declare_parameter('inference_frequency', 50.0)  # Hz
        self.declare_parameter(
            'urdf_path', get_package_share_directory("quadruped") + "/urdf/go2_description.urdf")
        self.declare_parameter(
            'package_dir', get_package_share_directory('quadruped'))

        model_path = self.get_parameter(
            'model_path').get_parameter_value().string_value
        self._load_policy(model_path)
        inference_frequency = self.get_parameter(
            'inference_frequency').get_parameter_value().double_value
        self._inference_period = 1.0 / inference_frequency
        urdf_path = self.get_parameter(
            'urdf_path').get_parameter_value().string_value
        package_dir = self.get_parameter(
            'package_dir').get_parameter_value().string_value
        robot = robot_loader.get_pin_robot_wrapper(urdf_path, package_dir)

        self.observation_subscriber = self.create_subscription(
            Float64MultiArray,
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

        self.inference_timer = self.create_timer(
            self._inference_period, self.inference_timer_callback)
        self.action_publisher = self.create_publisher(
            Float64MultiArray,
            '/action',
            QoSProfile(
                reliability=QoSReliabilityPolicy.BEST_EFFORT,
                history=QoSHistoryPolicy.KEEP_LAST,
                depth=1
            )
        )

        self._observation = None
        self._last_policy_output = None
        self._joint_states = None
        self._action_cfg = go2_action_config()
        self._action_cfg.residual_scale = 0.02
        self._trajectory_generators = [
            trajectory_generator.HybridFourDimTrajectoryGenerator(
                self._action_cfg.trajectory_generator_params, i)
            for i in range(4)
        ]
        self._phases = torch.zeros(1, 4)
        self._ik_solver = ik.InverseKinematicsSolver(
            robot_wrapper=robot,
            ee_name_list=['FL_foot', 'FR_foot', 'RL_foot', 'RR_foot'],
            rate=1.0
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

    def _compute_policy(self, obs: np.ndarray) -> np.ndarray:
        """
        Computes the action from the observation using the loaded policy.

        Args:
            obs (np.ndarray): The observation.

        Returns:
            np.ndarray: The action.
        """
        with torch.no_grad():
            obs = torch.from_numpy(obs).view(1, -1).float()  # type: ignore
            output = self.policy(obs).detach().view(-1).numpy()
        return output

    def _compute_joint_targets(self, policy_output: np.ndarray) -> np.ndarray:
        """
        Computes the target joint positions based on the action and current joint states.

        Args:
            action (np.ndarray): The action from the policy.
        Returns:
            np.ndarray: The target joint positions.
        """
        if self._joint_states is None:
            self.get_logger().warning('No joint state or base pose received yet.')
            return np.zeros(12)  # FIXME: use default pose

        reordered_positions, _, _ = message_processor.reorder_joint_states_to_numpy(
            self._joint_states,
            ['FL_hip_joint', 'FL_thigh_joint', 'FL_calf_joint',
             'FR_hip_joint', 'FR_thigh_joint', 'FR_calf_joint',
             'RL_hip_joint', 'RL_thigh_joint', 'RL_calf_joint',
             'RR_hip_joint', 'RR_thigh_joint', 'RR_calf_joint']
        )
        if reordered_positions is None:
            self.get_logger().error('Failed to reorder joint states.')
            return np.zeros(12)

        tg_args = torch.from_numpy(policy_output[:4]).view(1, -1).double()
        foot_target_positions = []
        for trajectory_generator_idx, trajectory_generator in enumerate(self._trajectory_generators):
            foot_target_position, phase = trajectory_generator.generate(
                tg_args, self._inference_period)
            foot_target_positions.append(foot_target_position)
            self._phases[:, trajectory_generator_idx] = phase

        # foot_target_positions = [[0.1934, 0.1465, -0.3], [0.1934, -0.1465, -0.3],[-0.2934, 0.1465, -0.3], [-0.2934, -0.1465, -0.3]] # FIXME: temporary for test
        joint_targets = np.zeros(12)
        for idx, foot in enumerate(['FL_foot', 'FR_foot', 'RL_foot', 'RR_foot']):
            try:
                joint_targets[idx * 3: (idx + 1) * 3] = self._ik_solver.solve_ik(
                    ee_name=foot,
                    ee_target_pos=torch.squeeze(foot_target_positions[idx]),
                    # ee_target_pos=foot_target_positions[idx],
                    curr_q=reordered_positions,
                )[idx * 3: (idx + 1) * 3] + policy_output[4 + idx * 3: 4 + (idx + 1) * 3] * self._action_cfg.residual_scale
            except Exception as e:
                self.get_logger().error(f'IK solver error for {foot}: {e}')

        return joint_targets

    def observation_callback(self, msg: Float64MultiArray):
        """
        Callback function for the observation subscriber.

        Args:
            msg (Float64MultiArray): The incoming observation message.
        """
        self._observation = np.array(msg.data)

    def joint_state_callback(self, msg: JointState):
        """
        Callback function for the joint state subscriber.

        Args:
            msg (JointState): The incoming joint state message.
        """
        self._joint_states = msg

    def inference_timer_callback(self):
        """
        Timer callback to perform inference and log the action.
        """
        if self.observation is not None:
            policy_output = self._compute_policy(self.observation)
            self._last_policy_output = policy_output
            final_action = self._compute_joint_targets(policy_output)
            action_msg = Float64MultiArray()
            action_msg.data = final_action.tolist()
            self.action_publisher.publish(action_msg)
        else:
            self.get_logger().warning('No observation received yet.')

    @property
    def observation(self):
        if self._observation is None:
            return None

        if self._last_policy_output is not None:
            self._observation[36:52] = self._last_policy_output[:]

        return self._observation[:52]   # FIXME: 之後要加上相位


def main():
    rclpy.init()
    node = InferenceNode()
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == '__main__':
    main()
