import numpy as np

import pinocchio as pin
import pink
from pink import solve_ik as pink_solve_ik, FrameTask
import qpsolvers
from loop_rate_limiters import RateLimiter


class InverseKinematicsSolver():
    def __init__(self, robot_wrapper: pin.RobotWrapper, ee_name_list: list[str], q_ref: np.ndarray | None = None, rate=50.0, solver="proxqp"):
        """
        Initialize the inverse kinematics solver.
        Args:
            urdf_filename (str): Path to the URDF file of the robot.
            package_dirs (list[str], optional): List of package directories for resolving URDF dependencies.
            q_ref (np.ndarray, optional): Reference joint configuration for the robot.
        """

        self._robot = robot_wrapper

        q_mid_range = (self._robot.model.lowerPositionLimit +
                       self._robot.model.upperPositionLimit) / 2.0
        q = q_ref if q_ref is not None else q_mid_range
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
        clipped_q = np.clip(
            curr_q,
            self._robot.model.lowerPositionLimit,
            self._robot.model.upperPositionLimit
        )
        self._configuration.update(clipped_q)

        # Iteratively solve for the joint configuration
        max_iterations = 50
        tolerance = 1e-3  # 1 mm

        for i in range(max_iterations):
            # 使用目前的腳關節計算要到達目標位置所需的關節速度, 目前腳關節角度存在 self._configuration.q
            velocity = pink_solve_ik(
                self._configuration,
                [task],
                dt,
                solver=self.solver,
                damping=1.0e-2,
            )
            self._configuration.integrate_inplace(velocity, dt)

            # 計算位置誤差
            current_pose = self._configuration.get_transform_frame_to_world(
                ee_name)
            position_error = target_pos - current_pose.translation
            position_error_norm = np.linalg.norm(position_error)

            if position_error_norm < tolerance:
                # print(
                #     f"IK converged for {ee_name} in {i + 1} iterations. "
                #     f"Error: {position_error_norm:.6f} m"
                # )
                break
            # This block executes if the loop completes without a break
            # elif i == max_iterations - 1 and position_error_norm > tolerance:
            #     print(
            #         f"Warning: IK for {ee_name} did not converge after {max_iterations} "
            #         f"iterations. Final error: {position_error_norm:.6f} m"
            #     )

        return self._configuration.q

    @property
    def configuration(self):
        return self._configuration

    @property
    def robot(self):
        return self._robot
