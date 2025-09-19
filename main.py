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

    configuration = pink.Configuration(robot.model, robot.data, q_ref)

    tasks = []
    foot_tasks = {}
    swing_legs = ["FL_foot", "RR_foot", "FR_foot", "RL_foot"]
    for foot in swing_legs:
        task = FrameTask(
            foot,
            position_cost=1.0,  # [cost] / [m]
            orientation_cost=0.0,  # [cost] / [rad]
        )
        tasks.append(task)
        foot_tasks[foot] = task

    # # Add a posture task to constrain the robot's body posture
    # posture_task = PostureTask(
    #     cost=1.0,  # [cost] / [rad]
    # )
    # tasks.append(posture_task)
    # Add a task to keep the base horizontal
    base_task = FrameTask(
        "base",
        position_cost=1.0,
        orientation_cost=1.0,
    )
    tasks.append(base_task)

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
    swing_legs = ["FL_foot", "RR_foot", "FR_foot", "RL_foot"]

    while True:
        # Determine which leg is swinging based on time
        swing_time = t % period
        phase = swing_time / (period / len(swing_legs))
        leg_index = int(phase)
        leg_phase = phase - leg_index
        swing_leg = swing_legs[leg_index]

        # Update foot targets
        for name, task in foot_tasks.items():
            target = task.transform_target_to_world
            initial_pos = initial_foot_positions[name]
            if name == swing_leg:
                # Swing leg moves up and down in a sine wave
                target.translation[0] = initial_pos[0]
                target.translation[1] = initial_pos[1]
                target.translation[2] = initial_pos[2] + swing_height * np.sin(
                    leg_phase * np.pi
                )
            else:
                # Other legs stay on the ground
                target.translation[:] = initial_pos

        velocity = solve_ik(
            configuration,
            tasks,
            dt,
            solver=solver,
        )
        configuration.integrate_inplace(velocity, dt)

        viz.display(configuration.q)
        rate.sleep()
        t += dt
