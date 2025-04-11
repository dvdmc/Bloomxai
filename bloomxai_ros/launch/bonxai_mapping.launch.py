import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.substitutions import Command
from launch_ros.actions import Node

# this is the function launch  system will look for
def generate_launch_description():

    # Current Package Name
    package = "bloomxai_ros"

    bloomxai_params = os.path.join(
        get_package_share_directory(package),
        'params',
        'bloomxai_params.yaml'
        )

    # Bonxai Server Node
    bloomxai_node = Node(
        package=package,
        executable='bloomxai_server_node',
        name='bloomxai_server_node',
        emulate_tty=True,
        parameters=[bloomxai_params],
        output="screen"
    )

    # Launch Nodes
    return LaunchDescription(
        [
            bloomxai_node,
        ]
    )
