from typing import Tuple
from dataclasses import MISSING, dataclass
import torch


@dataclass
class ActionCfg:
    @dataclass
    class TrajectoryGeneratorCfg:
        """Configuration for the trajectory generator used in PMTG."""

        leg_hip_positions: tuple[list[float], list[float],
                                 list[float], list[float]] = MISSING  # LF, RF, RL, RR
        """四條腿的髖關節相對於機身的位置, 用於計算轉向效果"""

        default_swing_duty_cycle: float = 0.5
        """Fixed swing duty cycle ratio. Defaults to 0.5."""

        step_height_limit: tuple[float, float] = (0.0, 0.2)
        """Step height limits (m). Defaults to (0.0, 0.2)."""

        # Frequency limits
        frequency_limit: tuple[float, float] = (1.0, 4.0)
        """Frequency limits (Hz). Defaults to (1.0, 4.0)."""

        # Step length limits
        step_length_x_limit: tuple[float, float] = (-0.4, 0.4)
        """X step length limits (m). Defaults to (-0.4, 0.4)."""

        step_length_y_limit: tuple[float, float] = (-0.2, 0.2)
        """Y step length limits (m). Defaults to (-0.2, 0.2)."""

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

    trajectory_generator_params: TrajectoryGeneratorCfg = MISSING  # type: ignore

    gain: float = 1.0
    """增益因子, 用於apply_action效果的強度"""

    residuals_limit: tuple[float, float] = (-0.1, 0.1)
    """關節位置殘差的限制範圍, 防止過大的調整"""


class HybridFourDimTrajectoryGenerator:
    """
    單條腿之混合控制軌跡生成器 (批次處理版本), 基於 CPG 核心參數。

    它接收一個 4 維的動作張量, shape 為 (batch_size, 4), 4個維度分別是:

    - 頻率 (frequency, f): 步態的頻率 (Hz)。
    - X 軸振幅 (amplitude_x, Ax): 控制前後步長。
    - Y 軸振幅 (amplitude_y, Ay): 控制側向步長。轉向可透過為左右腿設置不同的 Ay 實現。
    - Z 軸振幅 (amplitude_z, Az): 控制抬腿高度。

    擺動相占空比 (swing_duty_cycle) 固定。
    """

    def __init__(
        self,
        trajectory_generator_params: ActionCfg.TrajectoryGeneratorCfg,
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
        self.device = (
            torch.device(device) if device is not None else torch.device("cpu")
        )
        self.dtype = dtype
        self.trajectory_generator_params = trajectory_generator_params
        self.leg_index = leg_index

        # 從對應的腿索引取得參數 (視為基礎偏移量 Offset)
        self.default_foot_height = torch.as_tensor(
            trajectory_generator_params.foot_default_heights[leg_index],
            dtype=self.dtype,
            device=self.device,
        )
        self.default_y_offset = torch.as_tensor(
            trajectory_generator_params.leg_y_offsets[leg_index],
            dtype=self.dtype,
            device=self.device,
        )
        self.default_x_offset = torch.as_tensor(
            trajectory_generator_params.leg_x_offsets[leg_index],
            dtype=self.dtype,
            device=self.device,
        )

        # 相位 (初始化為 scalar tensor, 會在 generate 中根據 batch_size 自動擴展)
        # 這是實現腿間相位差 (Δφ) 的基礎
        self.phase = torch.tensor(
            trajectory_generator_params.phase_offsets[leg_index] % 1.0,
            device=self.device,
            dtype=self.dtype,
        )

        self.leg_hip_position = torch.as_tensor(
            trajectory_generator_params.leg_hip_positions[leg_index],
            dtype=self.dtype,
            device=self.device,
        )
        assert self.leg_hip_position.shape == (
            3,
        ), "leg_hip_position 必須是 shape (3,) 的向量"

        # 從配置中取得參數
        self.default_swing_duty_cycle = torch.as_tensor(
            trajectory_generator_params.default_swing_duty_cycle,
            dtype=self.dtype,
            device=self.device,
        )

    def _update_phase(self, frequency: torch.Tensor, dt: float | torch.Tensor):
        """根據頻率與時間步長更新此腿相位 (支援批次處理)。"""
        dt_t = torch.as_tensor(dt, dtype=self.dtype, device=self.device)
        # 使用 fmod 保持在 [0,1)
        self.phase = torch.fmod(self.phase + frequency * dt_t, 1.0)

    def generate(
        self, actions: torch.Tensor, dt: float | torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        計算單腿足端目標 (x, y, z)，支援批次處理。

        Args:
            actions (torch.Tensor): 來自 policy 的 CPG 調變參數張量, shape (batch_size, 4)。
                                    分別為 (frequency, amplitude_x, amplitude_y, amplitude_z)
            dt (float or torch.Tensor): 單步控制時間 (s)。可以是 scalar 或 shape (batch_size,)。

        Returns:
            torch.Tensor: 目標足端位置, shape (batch_size, 3) -> [[x1, y1, z1], [x2, y2, z2], ...]
            torch.Tensor: 當前相位, shape (batch_size,)
        """
        batch_size = actions.shape[0]

        # 檢查並在必要時擴展 self.phase 以匹配 batch_size
        if self.phase.numel() != batch_size:
            self.phase = self.phase.expand(batch_size).clone()

        # 1. 使用 tanh 將 CPG 參數從 (-inf, inf) 映射到 (-1, 1)
        actions_on_device = actions.to(self.device, self.dtype)
        frequency, amp_x, amp_y, amp_z = torch.unbind(actions_on_device, dim=1)

        # 2. 使用固定的占空比
        target_swing_duty_cycle = self.default_swing_duty_cycle
        target_stance_duty_cycle = 1.0 - target_swing_duty_cycle

        # 3. 更新相位並計算軌跡
        self._update_phase(frequency, dt)

        # --- 使用 torch.where 取代 if/else 邏輯 ---
        is_swing = self.phase < target_swing_duty_cycle

        # 為 is_swing=True 和 is_swing=False 兩種情況都計算 phase
        phase_in_swing = self.phase / target_swing_duty_cycle
        phase_in_stance = (
            self.phase - target_swing_duty_cycle
        ) / target_stance_duty_cycle

        # --- Z 軸軌跡 (由振幅 Az 控制) ---
        z_swing_offset = 0.5 * amp_z * \
            (1 - torch.cos(2 * torch.pi * phase_in_swing))
        z_stance_offset = torch.zeros_like(z_swing_offset)
        z_offset = torch.where(is_swing, z_swing_offset, z_stance_offset)
        # 最終 Z 軸位置 = 預設高度 (偏移量 O_z) + 軌跡
        z = self.default_foot_height + z_offset

        # --- X, Y 軸軌跡 (由振幅 Ax, Ay 控制) ---
        swing_multiplier = -0.5 * torch.cos(torch.pi * phase_in_swing)
        x_swing = amp_x * swing_multiplier
        y_swing = amp_y * swing_multiplier

        stance_multiplier = 0.5 * (1 - 2 * phase_in_stance)
        x_stance = amp_x * stance_multiplier
        y_stance = amp_y * stance_multiplier

        x_motion = torch.where(is_swing, x_swing, x_stance)
        y_motion = torch.where(is_swing, y_swing, y_stance)

        # 最終 X, Y 軸位置 = 預設偏移量 (O_x, O_y) + 軌跡
        x = self.default_x_offset + x_motion
        y = self.default_y_offset + y_motion

        # 注意：轉向 (Yaw) 效果應由上層控制器通過為左右腿提供不同的 `amplitude_y` 來實現，
        # 因此這裡不再單獨處理 `yaw_rate`。

        # 將 x, y, z 組合成 (batch_size, 3) 的張量
        foot_pos_rel_hip = torch.stack([x, y, z], dim=1)
        # 加上髖關節在基座標系下的位置，得到相對於基座標系的足端位置
        # 回傳足端位置以及相位，提供給觀測空間
        return (foot_pos_rel_hip + self.leg_hip_position, self.phase)


def go2_action_config():
    return ActionCfg(
        gain=1.0,
        trajectory_generator_params=ActionCfg.TrajectoryGeneratorCfg(
            leg_hip_positions=([0.1934, 0.0465, 0.0], [0.1934, -0.0465, 0.0],
                               [-0.1934, 0.0465, 0.0], [-0.1934, -0.0465, 0.0]),  # FL, FR, RL, RR
            foot_default_heights=(-0.3, -0.3, -0.32, -0.32),
            leg_y_offsets=(0.12, -0.12, 0.12, -0.12),
            leg_x_offsets=(0.02, 0.02, -0.05, -0.05),
        ),
    )
