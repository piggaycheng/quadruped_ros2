#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# SPDX-License-Identifier: Apache-2.0
# Copyright 2024 Ivan Domrachev, Simeon Nedelchev

"""Go2 squat with z-axis barrier."""

import numpy as np
import pinocchio as pin
import qpsolvers

import meshcat_shapes
import pink
from pink import solve_ik
from pink.barriers import PositionBarrier
from pink.tasks import FrameTask, PostureTask
from pink.visualization import start_meshcat_visualizer

from quadruped.pmtg.ik import InverseKinematicsSolver

try:
    from loop_rate_limiters import RateLimiter
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        "Examples use loop rate limiters, "
        "try `[conda|pip] install loop-rate-limiters`"
    ) from exc

urdf_filename = "./urdf/go2_description/urdf/go2_description.urdf"
package_dir = "./urdf/"


if __name__ == "__main__":
    root_joint = pin.JointModelFreeFlyer()
    robot = pin.RobotWrapper.BuildFromURDF(
        urdf_filename,
        package_dirs=package_dir,
        root_joint=root_joint,
    )
    viz = start_meshcat_visualizer(robot)
    swing_legs = ["FL_foot", "RR_foot", "FR_foot", "RL_foot"]
    q_ref = np.array(
        [
            -0.0,
            0.0,
            0.3,
            0.0,
            0.0,
            0.0,
            1.0,
            0.0,
            0.8,
            -1.57,
            0.0,
            0.8,
            -1.57,
            0.0,
            0.8,
            -1.57,
            0.0,
            0.8,
            -1.57,
        ]
    )
    ik_solver = InverseKinematicsSolver(
        robot, ee_name_list=swing_legs, base_name="base", q_ref=q_ref)

    configuration = pink.Configuration(robot.model, robot.data, q_ref)

    tasks = []
    foot_tasks = {}

    for foot in swing_legs:
        task = FrameTask(
            foot,
            position_cost=1.0,  # [cost] / [m]
            orientation_cost=0.0,  # [cost] / [rad]
        )
        tasks.append(task)
        foot_tasks[foot] = task

    for task in tasks:
        task.set_target_from_configuration(configuration)

    # Store initial foot positions
    initial_foot_positions = {
        name: task.transform_target_to_world.translation.copy()
        for name, task in foot_tasks.items()
    }

    viewer = viz.viewer
    opacity = 0.5  # Set the desired opacity level (0 transparent, 1 opaque)

    # Select QP solver
    solver = qpsolvers.available_solvers[0]
    if "osqp" in qpsolvers.available_solvers:
        solver = "osqp"

    rate = RateLimiter(frequency=200.0)
    dt = rate.period
    t = 0.0  # [s]
    period = 2.0  # Gait period in seconds
    omega = 2 * np.pi / period
    swing_height = 0.15  # Foot swing height in meters
    swing_legs = ["FL_foot", "FR_foot", "RL_foot", "RR_foot"]

    while True:
        # Determine which leg is swinging based on time
        swing_time = t % period
        phase = swing_time / (period / len(swing_legs))
        leg_index = int(phase)
        leg_phase = phase - leg_index
        swing_leg = swing_legs[leg_index]

        # Update foot targets
        for leg_name in swing_legs:
            initial_pos = initial_foot_positions[leg_name]
            if leg_name == swing_leg:
                q = ik_solver.solve_ik(
                    leg_name,
                    [initial_pos[0],
                     initial_pos[1],
                     initial_pos[2] + swing_height * np.sin(leg_phase * np.pi)],    # ee目標位置
                    ik_solver.configuration.q,      # 模擬傳入目前關節角度
                )
                ik_solver.configuration.update(q)       # 模擬更新目前關節角度

        viz.display(ik_solver.configuration.q)
        rate.sleep()
        t += dt
