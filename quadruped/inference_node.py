import torch
import numpy as np
import yaml
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
from std_msgs.msg import Float64MultiArray
from geometry_msgs.msg import Twist
from sensor_msgs.msg import JointState, Imu
from ament_index_python.packages import get_package_share_directory

from .pmtg import ik, trajectory_generator
from .pmtg.trajectory_generator import go2_action_config
from .utils import robot_loader, message_processor, action as action_utils, observation as observation_utils

import debugpy

debugpy.listen(5678)
print("Waiting for debugger attach...")
debugpy.wait_for_client()
print("Debugger attached")


class InferenceNode(Node):
    def __init__(self):
        super().__init__('inference_node')

        self.declare_parameter(
            'model_path', "resource/policies/go2_pmtg/policy.pt")
        self.declare_parameter(
            'env_yaml_path', "resource/policies/go2_pmtg/env.yaml")
        self.declare_parameter(
            'joints_order', ['joint1', 'joint2', 'joint3', 'joint4'])
        self.declare_parameter('inference_frequency', 50.0)  # Hz
        self.declare_parameter(
            'urdf_path', "resource/urdf/go2_description.urdf")
        self.declare_parameter(
            'package_dir', ".")

        model_path = f"{get_package_share_directory('quadruped')}/{self.get_parameter('model_path').get_parameter_value().string_value}"
        env_yaml_path = f"{get_package_share_directory('quadruped')}/{self.get_parameter('env_yaml_path').get_parameter_value().string_value}"
        self._joints_order = self.get_parameter(
            'joints_order').get_parameter_value().string_array_value
        self._env_config = self.load_env_yaml(env_yaml_path)
        self._joint_default_pos = robot_loader.construct_robot_default_joint_pos(
            env_joint_pos_regex=self._robot_default_joint_pos,
            joint_names=self._joints_order
        )
        self._load_policy(model_path)
        inference_frequency = self.get_parameter(
            'inference_frequency').get_parameter_value().double_value
        self._inference_period = 1.0 / inference_frequency
        urdf_path = f"{get_package_share_directory('quadruped')}/{self.get_parameter('urdf_path').get_parameter_value().string_value}"
        package_dir = f"{get_package_share_directory('quadruped')}/{self.get_parameter('package_dir').get_parameter_value().string_value}"
        robot = robot_loader.get_pin_robot_wrapper(urdf_path, package_dir)

        sensor_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1
        )

        reliable_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=5
        )

        self.observation_subscriber = self.create_subscription(
            Float64MultiArray,
            '/observation',
            self.observation_callback,
            sensor_qos
        )
        self.joint_state_subscriber = self.create_subscription(
            JointState,
            '/joint_states',
            self.joint_state_callback,
            sensor_qos
        )
        self.imu_subscriber = self.create_subscription(
            Imu,
            '/imu',
            self.imu_callback,
            sensor_qos
        )
        self.command_subscriber = self.create_subscription(
            Twist,
            '/command',
            self.command_callback,
            reliable_qos
        )

        self.inference_timer = self.create_timer(
            self._inference_period, self.inference_timer_callback)
        self.action_publisher = self.create_publisher(
            Float64MultiArray,
            '/action',
            sensor_qos
        )

        self._observation = np.zeros(72)
        self._last_policy_output = None
        self._joint_states = None
        self._imu_data = None
        self._command = None
        self._action_cfg = go2_action_config()

        # Action processing state
        self._processed_actions = np.zeros(20)
        self._current_fade = np.zeros(1)
        self._fade_speed = 0.05

        self._trajectory_generators = [
            trajectory_generator.HybridFourDimTrajectoryGenerator(
                self._action_cfg.trajectory_generator_params, i)
            for i in range(4)
        ]
        self._phases = torch.zeros(1, 4)
        self._ik_solver = ik.InverseKinematicsSolver(
            robot_wrapper=robot,
            ee_name_list=['FL_foot', 'FR_foot', 'RL_foot', 'RR_foot'],
            rate=50.0
        )
        self._joint_pos_des = None

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

    def load_env_yaml(self, yaml_path: str) -> dict:
        """
        Load environment configuration from a YAML file.

        Args:
            yaml_path (str): Path to the YAML file.
        Returns:
            dict: The loaded configuration.
        """
        with open(yaml_path, 'r') as file:
            config = yaml.unsafe_load(file)

        self._robot_default_joint_pos = config.get('scene', {}).get(
            'robot', {}).get('init_state', {}).get('joint_pos', {})

        return config

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

        # Apply tanh
        actions = np.tanh(policy_output)

        last_cpg_args = self._processed_actions[:8].copy()
        last_residuals = self._processed_actions[8:].copy()

        cpg_actions_raw = actions[:8]
        frequency = cpg_actions_raw[0]
        amp_x = cpg_actions_raw[1]
        amp_y = cpg_actions_raw[2]
        amp_z = cpg_actions_raw[3]
        offset_x = cpg_actions_raw[4]
        offset_y = cpg_actions_raw[5]
        offset_z = cpg_actions_raw[6]
        yaw_param = cpg_actions_raw[7]

        tg_params = self._action_cfg.trajectory_generator_params

        # Mapping
        processed_cpg_args = np.array([
            action_utils.tanh_post_process(
                frequency, tg_params.frequency_limit),
            action_utils.tanh_post_process(
                amp_x, tg_params.step_length_x_limit),
            action_utils.tanh_post_process(
                amp_y, tg_params.step_length_y_limit),
            action_utils.tanh_post_process(
                amp_z, tg_params.step_height_limit),
            action_utils.tanh_post_process(
                offset_x, tg_params.offset_x_limit),
            action_utils.tanh_post_process(
                offset_y, tg_params.offset_y_limit),
            action_utils.tanh_post_process(
                offset_z, tg_params.offset_z_limit),
            action_utils.tanh_post_process(
                yaw_param, tg_params.yaw_limit),
        ])

        # LPF (Filter)
        processed_cpg_args = (
            self._action_cfg.cpg_lpf_alpha * processed_cpg_args
            + (1 - self._action_cfg.cpg_lpf_alpha) * last_cpg_args
        )

        # Fade Factor
        if self._command is None:
            speed_norm = 0.0
        else:
            cmd_vel = np.array(
                [self._command.linear.x, self._command.linear.y])
            speed_norm = np.linalg.norm(cmd_vel)

        target_fade = 1.0 if speed_norm > self._action_cfg.command_threshold else 0.0

        diff = target_fade - self._current_fade
        step = np.clip(diff, -self._fade_speed, self._fade_speed)
        self._current_fade += step

        # Apply fade factor to Amps(1-3), Offsets(4-6), Yaw(7)
        params_to_fade = processed_cpg_args[1:8]
        processed_cpg_args[1:8] = params_to_fade * self._current_fade

        # Process residuals
        residuals_raw = actions[8:]
        processed_residuals = action_utils.tanh_post_process(
            residuals_raw, self._action_cfg.residuals_limit
        )

        # Apply LPF to residuals
        processed_residuals = (
            self._action_cfg.residuals_lpf_alpha * processed_residuals
            + (1 - self._action_cfg.residuals_lpf_alpha) * last_residuals
        )

        self._processed_actions = np.concatenate(
            [processed_cpg_args, processed_residuals]
        )

        tg_args = torch.from_numpy(processed_cpg_args).double().unsqueeze(0)

        foot_target_positions = []
        for trajectory_generator_idx, trajectory_generator in enumerate(self._trajectory_generators):
            foot_target_position, phase = trajectory_generator.generate(
                tg_args, self._inference_period)
            foot_target_positions.append(
                foot_target_position.detach().cpu().numpy().squeeze())
            self._phases[:, trajectory_generator_idx] = phase

        joint_targets = np.zeros(12)
        for idx, foot in enumerate(['FL_foot', 'FR_foot', 'RL_foot', 'RR_foot']):
            try:
                ik_joint_targets = self._ik_solver.solve_ik(
                    ee_name=foot,
                    ee_target_pos=foot_target_positions[idx],
                    curr_q=self.urdf_joint_pos,
                )[idx * 3: (idx + 1) * 3]

                processed_residual = processed_residuals[idx *
                                                         3: (idx + 1) * 3]

                joint_targets[idx * 3: (idx + 1) * 3] = ik_joint_targets + \
                    processed_residual
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

    def imu_callback(self, msg: Imu):
        """
        Callback function for the IMU subscriber.

        Args:
            msg (Imu): The incoming IMU message.
        """
        self._imu_data = msg

    def command_callback(self, msg: Twist):
        """
        Callback function for the command subscriber.

        Args:
            msg (Twist): The incoming command message.
        """
        self._command = msg

    def inference_timer_callback(self):
        """
        Timer callback to perform inference and log the action.
        """
        if self.observation is not None:
            policy_output = self._compute_policy(self.observation)
            self._last_policy_output = policy_output
            final_action = self._compute_joint_targets(policy_output)
            self._joint_pos_des = final_action
            action_msg = Float64MultiArray()
            action_msg.data = final_action.tolist()
            self.action_publisher.publish(action_msg)
        else:
            self.get_logger().warning('No observation received yet.')

    @property
    def observation(self):
        if self._observation is None:
            return None

        self._observation[:3] = self.ang_vel[:]
        self._observation[3:6] = self.projected_gravity[:]
        self._observation[6:9] = self.command[:]
        self._observation[9:21] = self.joints_states_pos_rel[:]
        self._observation[21:33] = self.joint_states.velocity[:]
        self._observation[33:49] = self.last_policy_output[:]
        self._observation[49:57] = self.phase_sin_cos[:]
        self._observation[57:69] = self.joint_pos_des[:]
        self._observation[69:72] = self.lin_acc[:]

        return self._observation

    @property
    def last_policy_output(self):
        if self._last_policy_output is None:
            return np.zeros(16)
        return self._last_policy_output

    @property
    def phase_sin_cos(self):
        if self._phases is None:
            return np.zeros(8)
        sin_phases = torch.sin(2 * np.pi * self._phases)
        cos_phases = torch.cos(2 * np.pi * self._phases)
        # Stack, flatten, and convert to numpy array of shape (8,)
        return torch.stack([sin_phases, cos_phases], dim=2).view(-1).numpy()

    @property
    def joint_pos_des(self):
        if self._joint_pos_des is None:
            return np.zeros(12)
        return self._joint_pos_des

    @property
    def projected_gravity(self):
        if self._imu_data is None:
            return np.zeros(3)
        orientation = self._imu_data.orientation
        quat = np.array([orientation.w, orientation.x,
                        orientation.y, orientation.z])
        return observation_utils.compute_projected_gravity(quat)

    @property
    def lin_acc(self):
        if self._imu_data is None:
            return np.zeros(3)
        linear_acceleration = self._imu_data.linear_acceleration
        return np.array([linear_acceleration.x,
                         linear_acceleration.y,
                         linear_acceleration.z])

    @property
    def ang_vel(self):
        if self._imu_data is None:
            return np.zeros(3)
        angular_velocity = self._imu_data.angular_velocity
        return np.array([angular_velocity.x,
                         angular_velocity.y,
                         angular_velocity.z])

    @property
    def command(self):
        if self._command is None:
            return np.zeros(3)
        return np.array([self._command.linear.x, self._command.linear.y, self._command.angular.z])

    @property
    def is_moving(self):
        if self._command is None:
            return False
        linear_speed = np.sqrt(self._command.linear.x **
                               2 + self._command.linear.y**2)
        angular_speed = abs(self._command.angular.z)
        return linear_speed > self._action_cfg.command_threshold or angular_speed > self._action_cfg.command_threshold

    @property
    def joint_states(self):
        if self._joint_states is None:
            joint_state = JointState()
            joint_state.name = self._joints_order
            joint_state.position = [0.0] * 12
            joint_state.velocity = [0.0] * 12
            return joint_state

        return self._joint_states

    @property
    def joints_states_pos_rel(self):
        if self.joint_states is None:
            return np.zeros(12)
        return self.joint_states.position - np.array(
            [self._joint_default_pos[name] for name in self.joint_states.name])

    @property
    def urdf_joint_pos(self):
        reordered_positions, _, _ = message_processor.reorder_joint_states_to_numpy(
            self.joint_states,
            ['FL_hip_joint', 'FL_thigh_joint', 'FL_calf_joint',
             'FR_hip_joint', 'FR_thigh_joint', 'FR_calf_joint',
             'RL_hip_joint', 'RL_thigh_joint', 'RL_calf_joint',
             'RR_hip_joint', 'RR_thigh_joint', 'RR_calf_joint']
        )
        if reordered_positions is None:
            self.get_logger().error('Failed to reorder joint states.')
            return np.zeros(12)

        return reordered_positions


def main():
    rclpy.init()
    node = InferenceNode()
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == '__main__':
    main()
