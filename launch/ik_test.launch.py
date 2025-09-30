from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        Node(
            package='quadruped',
            executable='ik_test_node',
            name='ik_test_node',
            output='screen',
            parameters=[],
            remappings=[
                # You can add remappings here if needed
                # ('/joint_states', '/your_joint_states_topic'),
            ]
        )
    ])