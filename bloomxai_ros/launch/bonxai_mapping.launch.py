from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    config_file = LaunchConfiguration('config_file')

    return LaunchDescription([
        # Declare launch argument
        DeclareLaunchArgument(
            'config_file',
            default_value=PathJoinSubstitution([
                FindPackageShare('bloomxai_ros'),
                'cfg',
                'bloomxai.yaml'
            ]),
            description='Path to the configuration YAML file.'
        ),

        # Node definition
        Node(
            package='bloomxai_ros',
            executable='bloomxai_server_node',
            name='bloomxai_server',
            parameters=[config_file],
            remappings=[
                ('cloud_in', 'pointcloud')
            ],
            arguments=[],
            output='screen'
        ),
    ])
