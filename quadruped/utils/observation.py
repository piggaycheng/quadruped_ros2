import numpy as np


def compute_projected_gravity(orientation: np.ndarray) -> np.ndarray:
    """
    Compute the gravity vector projected by the robot's orientation.
    Args:
        orientation (np.ndarray): The orientation quaternion [w, x, y, z].
    Returns:
        np.ndarray: The gravity vector in the robot's local frame.
    """
    w, x, y, z = orientation
    return np.array([
        2 * (x * z - w * y),
        2 * (y * z + w * x),
        w * w - x * x - y * y + z * z
    ])
