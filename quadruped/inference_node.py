import torch
import numpy as np
from rclpy import init, spin, shutdown, Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
from std_msgs.msg import Float32MultiArray


class InferenceNode(Node):
    def __init__(self):
        super().__init__('inference_node')

        self.declare_parameter('model_path', 'path/to/your/model.pt')
        model_path = self.get_parameter(
            'model_path').get_parameter_value().string_value
        self._load_policy(model_path)

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
        self.inference_timer = self.create_timer(
            0.1, self.inference_timer_callback)
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

    def observation_callback(self, msg: Float32MultiArray):
        """
        Callback function for the observation subscriber.

        Args:
            msg (Float32MultiArray): The incoming observation message.
        """
        self._observation = np.array(msg.data)

    def inference_timer_callback(self):
        """
        Timer callback to perform inference and log the action.
        """
        if self._observation is not None:
            action = self._compute_action(self._observation)
            action_msg = Float32MultiArray()
            action_msg.data = action.tolist()
            self.action_publisher.publish(action_msg)
        else:
            self.get_logger().warning('No observation received yet.')


if __name__ == '__main__':
    init()
    node = InferenceNode()
    spin(node)
    shutdown()
