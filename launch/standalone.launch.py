
import os
from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python import get_package_share_directory


def generate_launch_description():
    # 找到您的參數檔案路徑
    params_file = os.path.join(
        get_package_share_directory('quadruped'),
        'config',
        'params.yaml'
    )

    inference_node = Node(
        package='quadruped',
        executable='inference_node',
        name='inference_node',
        output='screen',
        parameters=[params_file],
    )

    return LaunchDescription([inference_node])
