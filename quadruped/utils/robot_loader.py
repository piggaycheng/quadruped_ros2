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
