from typing import Tuple
from dataclasses import MISSING
import torch


class TrajectoryGeneratorCfg:
    stance_vx_scale: float = 1.0
    """Scale factor for the forward velocity command. Defaults to 1.0."""
    stance_vy_scale: float = 1.0
    """Scale factor for the lateral velocity command. Defaults to 1.0."""
    yaw_rate_scale: float = 1.0
    """Scale factor for the yaw rate command. Defaults to 1.0."""
    step_height_scale: float = 1.0
    """Scale factor for the step height command. Defaults to 1.0."""

    # Frequency and duty cycle parameters
    base_frequency: float = 1.5
    """Base frequency for gait generation (Hz). Defaults to 1.5."""
    velocity_to_freq_gain: float = 0.8
    """Gain for converting velocity to additional frequency. Defaults to 0.8."""
    default_swing_duty_cycle: float = 0.5
    """Fixed swing duty cycle ratio. Defaults to 0.5."""

    # Numerical stability
    eps: float = 1e-6
    """Small value to avoid division by zero. Defaults to 1e-6."""

    # Velocity limits
    stance_vx_limit: tuple[float, float] = (-0.8, 0.8)
    """Forward velocity limits (m/s). Defaults to (-0.8, 0.8)."""
    stance_vy_limit: tuple[float, float] = (-0.5, 0.5)
    """Lateral velocity limits (m/s). Defaults to (-0.5, 0.5)."""
    yaw_rate_limit: tuple[float, float] = (-1.5, 1.5)
    """Yaw rate limits (rad/s). Defaults to (-1.5, 1.5)."""
    step_height_limit: tuple[float, float] = (0.02, 0.2)
    """Step height limits (m). Defaults to (0.02, 0.2)."""

    # Frequency limits
    frequency_limit: tuple[float, float] = (1.0, 4.0)
    """Frequency limits (Hz). Defaults to (1.0, 4.0)."""

    # Step length limits
    step_length_limit: tuple[float, float] = (-0.3, 0.3)
    """Step length limits (m). Defaults to (-0.3, 0.3)."""

    foot_default_heights: tuple[float, float, float, float] = (
        0.0, 0.0, 0.0, 0.0)  # FL, FR, RL, RR
    """預設的腳部高度, 用於計算Z軸位置"""

    leg_y_offsets: tuple[float, float, float, float] = (
        0.0, 0.0, 0.0, 0.0)  # FL, FR, RL, RR
    """四條腿的Y軸預設偏移量, 用於計算Y軸位置"""

    leg_x_offsets: tuple[float, float, float, float] = (
        0.0, 0.0, 0.0, 0.0)  # FL, FR, RL, RR
    """四條腿的X軸預設偏移量, 用於計算X軸位置"""

    phase_offsets: tuple[float, float, float, float] = (
        0.0, 0.5, 0.5, 0.0)  # LF, RF, RL, RR
    """四條腿的相位偏移量, 以實現對角步態"""

    leg_hip_positions: tuple[list[float], list[float],
                             list[float], list[float]] = MISSING  # LF, RF, RL, RR
    """四條腿的髖關節相對於機身的位置, 用於計算轉向效果"""


class HybridFourDimTrajectoryGenerator:
    """
    單條腿之混合控制軌跡生成器 (批次處理版本)。

    它接收一個 2 維的完整動作張量, shape 為 (batch_size, 4), 4個維度分別是:

    - 前進速度 (stance_vx)
    - 側向速度 (stance_vy)
    - 轉向角速度 (yaw_rotation_rate)
    - 抬腿高度 (step_height)

    步頻 (frequency) 會根據期望速度自動調整，而擺動相占空比 (swing_duty_cycle) 則固定。
    """

    def __init__(self,
                 trajectory_generator_params: TrajectoryGeneratorCfg,
                 leg_index: int,
                 device: torch.device | str | None = None,
                 dtype: torch.dtype = torch.float32,
                 ):
        """
        初始化單腿軌跡生成器。

        Args:
            trajectory_generator_params: 軌跡生成器參數配置
            leg_index (int): 腿的索引 (0=FL, 1=FR, 2=RL, 3=RR)
            device: 計算設備
            dtype: 數據類型
        """
        self.device = torch.device(
            device) if device is not None else torch.device('cpu')
        self.dtype = dtype
        self.trajectory_generator_params = trajectory_generator_params
        self.leg_index = leg_index

        # 從對應的腿索引取得參數
        self.default_foot_height = torch.as_tensor(
            trajectory_generator_params.foot_default_heights[leg_index], dtype=self.dtype, device=self.device)
        self.default_y_offset = torch.as_tensor(
            trajectory_generator_params.leg_y_offsets[leg_index], dtype=self.dtype, device=self.device)
        self.default_x_offset = torch.as_tensor(
            trajectory_generator_params.leg_x_offsets[leg_index], dtype=self.dtype, device=self.device)

        # 相位 (初始化為 scalar tensor, 會在 generate 中根據 batch_size 自動擴展)
        self.phase = torch.tensor(
            trajectory_generator_params.phase_offsets[leg_index] % 1.0, device=self.device, dtype=self.dtype)

        self.leg_hip_position = torch.as_tensor(
            trajectory_generator_params.leg_hip_positions[leg_index], dtype=self.dtype, device=self.device)
        assert self.leg_hip_position.shape == (
            3,), "leg_hip_position 必須是 shape (3,) 的向量"

        # 從配置中取得參數
        self.base_frequency = torch.as_tensor(
            trajectory_generator_params.base_frequency, dtype=self.dtype, device=self.device)
        self.velocity_to_freq_gain = torch.as_tensor(
            trajectory_generator_params.velocity_to_freq_gain, dtype=self.dtype, device=self.device)
        self.default_swing_duty_cycle = torch.as_tensor(
            trajectory_generator_params.default_swing_duty_cycle, dtype=self.dtype, device=self.device)
        self.eps = torch.as_tensor(
            trajectory_generator_params.eps, dtype=self.dtype, device=self.device)

    def _update_phase(self, frequency: torch.Tensor, dt: float | torch.Tensor):
        """根據頻率與時間步長更新此腿相位 (支援批次處理)。"""
        dt_t = torch.as_tensor(dt, dtype=self.dtype, device=self.device)
        # 使用 fmod 保持在 [0,1)
        self.phase = torch.fmod(self.phase + frequency * dt_t, 1.0)

    def generate(self, actions: torch.Tensor, dt: float | torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        計算單腿足端目標 (x, y, z)，支援批次處理。

        Args:
            actions (torch.Tensor): 來自 policy 的調變參數張量, shape (batch_size, 4)。
            dt (float or torch.Tensor): 單步控制時間 (s)。可以是 scalar 或 shape (batch_size,)。

        Returns:
            torch.Tensor: 目標足端位置, shape (batch_size, 3) -> [[x1, y1, z1], [x2, y2, z2], ...]
        """
        batch_size = actions.shape[0]

        # 檢查並在必要時擴展 self.phase 以匹配 batch_size
        if self.phase.numel() != batch_size:
            # 使用第一個元素的值進行擴展，以保持一致的初始相位
            self.phase = self.phase.expand(batch_size).clone()

        # 1. 讀取 4 維參數並裁剪 (從 N,4 張量中分離)
        actions_on_device = actions.to(self.device, self.dtype)
        stance_vx, stance_vy, yaw_rate, step_height = torch.unbind(
            actions_on_device, dim=1)

        target_stance_vx = (stance_vx * self.trajectory_generator_params.stance_vx_scale).clamp(
            self.trajectory_generator_params.stance_vx_limit[0],
            self.trajectory_generator_params.stance_vx_limit[1]
        )
        target_stance_vy = (stance_vy * self.trajectory_generator_params.stance_vy_scale).clamp(
            self.trajectory_generator_params.stance_vy_limit[0],
            self.trajectory_generator_params.stance_vy_limit[1]
        )
        target_yaw_rate = (yaw_rate * self.trajectory_generator_params.yaw_rate_scale).clamp(
            self.trajectory_generator_params.yaw_rate_limit[0],
            self.trajectory_generator_params.yaw_rate_limit[1]
        )
        target_step_height = (step_height * self.trajectory_generator_params.step_height_scale).clamp(
            self.trajectory_generator_params.step_height_limit[0],
            self.trajectory_generator_params.step_height_limit[1]
        )

        # 2. 自動推算步頻
        linear_speed = torch.sqrt(target_stance_vx**2 + target_stance_vy**2)
        target_frequency = (self.base_frequency + self.velocity_to_freq_gain * linear_speed).clamp(
            self.trajectory_generator_params.frequency_limit[0],
            self.trajectory_generator_params.frequency_limit[1]
        )

        # 3. 使用固定的占空比
        target_swing_duty_cycle = self.default_swing_duty_cycle
        target_stance_duty_cycle = 1.0 - target_swing_duty_cycle

        # 4. 推導步幅
        # 避免除以零
        stance_duration = torch.where(
            target_frequency < self.eps,
            torch.zeros_like(target_frequency),
            target_stance_duty_cycle / target_frequency,
        )

        target_step_length_x = torch.clip(
            target_stance_vx * stance_duration,
            self.trajectory_generator_params.step_length_limit[0],
            self.trajectory_generator_params.step_length_limit[1]
        )
        target_step_length_y = torch.clip(
            target_stance_vy * stance_duration,
            self.trajectory_generator_params.step_length_limit[0],
            self.trajectory_generator_params.step_length_limit[1]
        )

        # 5. 更新相位並計算軌跡
        self._update_phase(target_frequency, dt)

        # --- 使用 torch.where 取代 if/else 邏輯 ---
        is_swing = self.phase < target_swing_duty_cycle

        # 為 is_swing=True 和 is_swing=False 兩種情況都計算 phase
        phase_in_swing = self.phase / target_swing_duty_cycle
        phase_in_stance = (
            self.phase - target_swing_duty_cycle) / target_stance_duty_cycle

        # --- Z 軸軌跡 ---
        z_swing_offset = 0.5 * target_step_height * \
            (1 - torch.cos(2 * torch.pi * phase_in_swing))
        z_stance_offset = torch.zeros_like(z_swing_offset)
        z_offset = torch.where(is_swing, z_swing_offset, z_stance_offset)
        # 最終 Z 軸位置 = 預設高度 + 位移
        z = self.default_foot_height + z_offset

        # --- X, Y 軸軌跡 (不含 yaw) ---
        swing_multiplier = -0.5 * torch.cos(torch.pi * phase_in_swing)
        x_swing = target_step_length_x * swing_multiplier
        y_swing = target_step_length_y * swing_multiplier

        stance_multiplier = 0.5 * (1 - 2 * phase_in_stance)
        x_stance = target_step_length_x * stance_multiplier
        y_stance = target_step_length_y * stance_multiplier

        x_motion = torch.where(is_swing, x_swing, x_stance)
        y_motion = torch.where(is_swing, y_swing, y_stance)

        x = self.default_x_offset + x_motion
        y = self.default_y_offset + y_motion

        # --- Yaw 效應 (僅在支撐相且頻率不為零時加入) ---
        apply_yaw_effect = (~is_swing) & (target_frequency > self.eps)

        # 預先計算 yaw 效應 (broadcasting 會自動處理)
        # 修正：將位移計算與 stance_duration 關聯，以符合物理模型
        # scale 因子 (1 - 2 * phase_in_stance) 會將位移從 +effect 掃描到 -effect，
        # 總位移是 effect 的兩倍。因此 effect 應為總位移的一半。
        total_displacement_yaw_x = - \
            self.leg_hip_position[1] * target_yaw_rate * stance_duration
        total_displacement_yaw_y = self.leg_hip_position[0] * \
            target_yaw_rate * stance_duration

        yaw_effect_x = 0.5 * total_displacement_yaw_x
        yaw_effect_y = 0.5 * total_displacement_yaw_y

        scale = (1 - 2 * phase_in_stance)

        # 僅在滿足條件時增加 yaw 效應
        x = torch.where(apply_yaw_effect, x + yaw_effect_x * scale, x)
        y = torch.where(apply_yaw_effect, y + yaw_effect_y * scale, y)

        # 將 x, y, z 組合成 (batch_size, 3) 的張量
        foot_pos_rel_hip = torch.stack([x, y, z], dim=1)
        # 加上髖關節在基座標系下的位置，得到相對於基座標系的足端位置
        # 回傳足端位置以及相位，提供給觀測空間
        return (foot_pos_rel_hip + self.leg_hip_position, self.phase)
