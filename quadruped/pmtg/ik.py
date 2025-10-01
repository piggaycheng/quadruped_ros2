import numpy as np

import pinocchio as pin
import pink
from pink import solve_ik, FrameTask
import qpsolvers
from loop_rate_limiters import RateLimiter


class InverseKinematicsSolver():
    def __init__(self, robot_wrapper: pin.RobotWrapper, ee_name_list: list[str], q_ref: np.ndarray | None = None, rate=50.0, solver="osqp"):
        """
        Initialize the inverse kinematics solver.
        Args:
            urdf_filename (str): Path to the URDF file of the robot.
            package_dirs (list[str], optional): List of package directories for resolving URDF dependencies.
            q_ref (np.ndarray, optional): Reference joint configuration for the robot.
        """

        self._robot = robot_wrapper

        q = q_ref if q_ref is not None else self._robot.q0
        self._configuration = pink.Configuration(
            self._robot.model, self._robot.data, q)

        self.rate_limiter = RateLimiter(rate)

        self.solver = qpsolvers.available_solvers[0]
        if solver in qpsolvers.available_solvers:
            self.solver = solver
        else:
            print(
                f"Warning: {solver} is not available. Using {self.solver} instead.")

        self.task_dict = {}

        for ee_name in ee_name_list:
            task = FrameTask(
                ee_name,
                position_cost=1.0,  # [cost] / [m]
                orientation_cost=0.0,  # [cost] / [rad]
            )
            self.task_dict[ee_name] = task

        for task in self.task_dict.values():
            task.set_target_from_configuration(self._configuration)

    def solve_ik(self, ee_name, ee_target_pos, curr_q) -> np.ndarray:
        """
        Solve the inverse kinematics problem to find the next joint configuration.

        Returns:
            np.ndarray: The next joint configuration.
        """
        dt = self.rate_limiter.period

        target_rot = np.identity(3)
        target_pos = np.array(ee_target_pos)
        target_pose = pin.SE3(target_rot, target_pos)  # type: ignore
        task = self.task_dict[ee_name]
        task.set_target(target_pose)

        # 更新目前的關節角度
        self._configuration.update(curr_q)

        # 使用目前的腳關節計算要到達目標位置所需的關節速度, 目前腳關節角度存在 self._configuration.q
        velocity = solve_ik(
            self._configuration,
            [task],
            dt,
            solver=self.solver,
        )

        return self._configuration.integrate(velocity, dt)

    @property
    def configuration(self):
        return self._configuration

    @property
    def robot(self):
        return self._robot
