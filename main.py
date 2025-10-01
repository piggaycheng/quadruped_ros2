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

urdf_filename = "./resource/urdf/go2_description.urdf"
# Update this path to your installed package share directory
package_dir = "/home/yu/ros2_workspace/install/quadruped/share"


if __name__ == "__main__":
    """
    以configuration模擬外部isaacsim中儲存的關節狀態, 
    ik_solver單純用來計算下一個關節角度,
    solve_ik傳入目前關節角度, 回傳下一個關節角度,
    外部configuration(isaacsim中的關節狀態)進行更新
    """

    robot = pin.RobotWrapper.BuildFromURDF(
        urdf_filename,
        package_dirs=package_dir,
    )
    viz = start_meshcat_visualizer(robot)
    swing_legs = ["FL_foot", "RR_foot", "FR_foot", "RL_foot"]
    q_ref = np.array([0.0, 0.8, -1.57, 0.0, 0.8, -1.57,
                     0.0, 0.8, -1.57, 0.0, 0.8, -1.57])
    configuration = pink.Configuration(robot.model, robot.data, q_ref)
    ik_solver = InverseKinematicsSolver(
        robot, ee_name_list=swing_legs, q_ref=None
    )

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

    rate = RateLimiter(frequency=50.0)
    dt = rate.period
    t = 0.0  # [s]
    period = 2.0  # Gait period in seconds
    omega = 2 * np.pi / period
    swing_height = 0.15  # Foot swing height in meters
    swing_leg_pairs = [["FL_foot", "RR_foot"], ["FR_foot", "RL_foot"]]

    while True:
        # Determine which pair of legs is swinging based on time
        swing_time = t % period
        phase = swing_time / (period / len(swing_leg_pairs))
        pair_index = int(phase)
        leg_phase = phase - pair_index
        swing_pair = swing_leg_pairs[pair_index]

        # Update foot targets
        q_current = configuration.q
        for leg_name in swing_legs:
            initial_pos = initial_foot_positions[leg_name]
            if leg_name in swing_pair:
                # This leg is swinging
                target_pos = [
                    initial_pos[0],
                    initial_pos[1],
                    initial_pos[2] + swing_height * np.sin(leg_phase * np.pi)
                ]
                q_current = ik_solver.solve_ik(
                    leg_name,
                    target_pos,    # ee目標位置
                    q_current,      # 模擬傳入目前關節角度
                )

        # After calculating all IK for the swinging legs, update the configuration once
        configuration.update(q_current)       # 模擬更新目前關節角度

        viz.display(configuration.q)
        rate.sleep()
        t += dt
