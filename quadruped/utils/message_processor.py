import numpy as np
from sensor_msgs.msg import JointState


def reorder_joint_states_to_numpy(
    msg: JointState,
    desired_order: list[str]
) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None]:
    """
    將 JointState 訊息按照指定的順序重新排列，並將其轉換為 NumPy 陣列。

    Args:
        msg (JointState): 從 /joint_states 收到的原始訊息。
        desired_order (list[str]): 你期望的關節名稱順序列表。

    Returns:
        tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None]:
        一個包含三個元素的元組，分別是排序後的 position, velocity, effort 的 NumPy 陣列。
        如果原始訊息中缺少某個欄位(例如 velocity)，則對應的返回值为 None。
        如果無法滿足期望的順序(例如缺少某個關節)，會引發一個 ValueError。
    """
    # 步驟 1: 建立一個從關節名稱到其在原始訊息中索引的映射
    # 這非常高效，只需遍歷一次原始資料
    name_to_index_map = {name: i for i, name in enumerate(msg.name)}

    # 檢查所有期望的關節是否存在於收到的訊息中
    if not all(name in name_to_index_map for name in desired_order):
        missing_joints = [
            name for name in desired_order if name not in name_to_index_map]
        raise ValueError(f"收到的 JointState 訊息中缺少以下關節: {missing_joints}")

    # 步驟 2: 根據期望的順序，使用映射來找到原始索引
    # list comprehension 提供了非常簡潔高效的寫法
    ordered_indices = [name_to_index_map[name] for name in desired_order]

    # 步驟 3: 使用排好序的索引來提取資料並轉換為 NumPy 陣列
    # 同時檢查原始訊息中是否有 position, velocity, effort 欄位

    ordered_positions = np.array(
        [msg.position[i] for i in ordered_indices]) if msg.position else None

    ordered_velocities = np.array(
        [msg.velocity[i] for i in ordered_indices]) if msg.velocity else None

    ordered_efforts = np.array([msg.effort[i]
                               for i in ordered_indices]) if msg.effort else None

    return ordered_positions, ordered_velocities, ordered_efforts
