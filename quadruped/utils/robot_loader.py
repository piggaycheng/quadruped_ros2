from pinocchio import RobotWrapper


def get_pin_robot_wrapper(urdf_filename: str, package_dirs: list[str] | str | None = None, root_joint=None) -> RobotWrapper:
    """
    Load a robot from a URDF file and return a Pinocchio RobotWrapper.
    Args:
        urdf_filename (str): Path to the URDF file of the robot.
        package_dirs (list[str] or str, optional): List of package directories for resolving URDF dependencies.
        root_joint (pin.JointModel, optional): The root joint model for the robot.
    Returns:
        RobotWrapper: The loaded robot wrapped in a Pinocchio RobotWrapper.
    """
    if package_dirs is None:
        package_dirs = []
    elif isinstance(package_dirs, str):
        package_dirs = [package_dirs]

    robot = RobotWrapper.BuildFromURDF(
        urdf_filename,
        package_dirs=package_dirs,
        root_joint=root_joint,
    )

    return robot


def construct_robot_default_joint_pos(env_joint_pos_regex, joint_names):
    """
    Construct a dictionary of default joint positions for the robot based on regex patterns.
    Args:
        env_joint_pos_regex (dict): A dictionary where keys are regex patterns and values are joint positions.
        joint_names (list[str]): List of joint names in the robot.
    Returns:
        dict: A dictionary mapping joint names to their default positions.
    """
    import re

    robot_default_joint_pos = {}
    for pattern, position in env_joint_pos_regex.items():
        regex = re.compile(pattern)
        for joint_name in joint_names:
            if regex.match(joint_name):
                robot_default_joint_pos[joint_name] = position

    return robot_default_joint_pos
