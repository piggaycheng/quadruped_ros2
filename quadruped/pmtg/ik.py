import numpy as np

import pinocchio as pin
import pink
from pink import solve_ik, FrameTask
import qpsolvers
from loop_rate_limiters import RateLimiter


class InverseKinematicsSolver():
    def __init__(self, robot_wrapper: pin.RobotWrapper, ee_name_list: list[str], q_ref: np.ndarray | None, rate=200.0, solver="osqp"):
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

        self.tasks = []
        self.ee_tasks = {}

        for ee_name in ee_name_list:
            task = FrameTask(
                ee_name,
                position_cost=1.0,  # [cost] / [m]
                orientation_cost=0.0,  # [cost] / [rad]
            )
            self.tasks.append(task)
            self.ee_tasks[ee_name] = task

        self.base_task = FrameTask(
            "base",
            position_cost=1.0,
            orientation_cost=1.0,
        )
        self.tasks.append(self.base_task)

        for task in self.tasks:
            task.set_target_from_configuration(self._configuration)

        # Store initial end-effector positions
        initial_ee_positions = {
            name: task.transform_target_to_world.translation.copy()
            for name, task in self.ee_tasks.items()
        }

    def solve_ik(self, ee_name, ee_target_pos) -> np.ndarray:
        """
        Solve the inverse kinematics problem to find the next joint configuration.

        Returns:
            np.ndarray: The next joint configuration.
        """

        dt = self.rate_limiter.period

        task = self.ee_tasks[ee_name]
        task.transform_target_to_world.translation[:] = ee_target_pos

        velocity = solve_ik(
            self._configuration,
            [self.ee_tasks[ee_name], self.base_task],
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