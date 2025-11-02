import numpy as np


def compute_projected_gravity(orientation: np.ndarray) -> np.ndarray:
    """
    Compute the gravity vector projected by the robot's orientation.
    Assume gravity vector in world frame is [0, 0, -1].
    
    Args:
        orientation (np.ndarray): The orientation quaternion [w, x, y, z].
    Returns:
        np.ndarray: The gravity vector in the robot's local frame.
    """
    gravity = np.array([0, 0, -1])
    w, x, y, z = orientation
    # Create a rotation matrix from the quaternion
    # This matrix transforms from body frame to world frame
    # To transform the world gravity vector to the body frame, we need its transpose
    rotation_matrix = np.array([
        [1 - 2 * (y**2 + z**2), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x**2 + z**2), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x**2 + y**2)]
    ])
    
    # The projected gravity in the robot's local frame is the rotation of the
    # world gravity vector by the inverse (transpose) of the orientation.
    projected_gravity = rotation_matrix.T.dot(gravity)
    return projected_gravity
    
