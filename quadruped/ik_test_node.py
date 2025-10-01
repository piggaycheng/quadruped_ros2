#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# SPDX-License-Identifier: Apache-2.0
# Copyright 2024 Ivan Domrachev, Simeon Nedelchev

"""ROS2 node for quadruped IK testing with joint state subscription."""

import debugpy
import numpy as np
import pinocchio as pin
import qpsolvers

import meshcat_shapes
import pink
from pink import solve_ik, Configuration
from pink.barriers import PositionBarrier
from pink.tasks import FrameTask
from pink.visualization import start_meshcat_visualizer

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float32MultiArray
from ament_index_python.packages import get_package_share_directory

from quadruped.pmtg.ik import InverseKinematicsSolver

try:
    from loop_rate_limiters import RateLimiter
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        "Examples use loop rate limiters, "
        "try `[conda|pip] install loop-rate-limiters`"
    ) from exc


debugpy.listen(5678)
print("Waiting for debugger attach...")
debugpy.wait_for_client()
print("Debugger attached")


class IKTestNode(Node):
    """ROS2 node for testing inverse kinematics with joint state feedback."""

    frequency = 50.0  # Hz
    target_configuration: Configuration | None = None
    current_joint_states: JointState | None = None

    def __init__(self):
        super().__init__('ik_test_node')

        # Initialize robot parameters
        self.urdf_filename = get_package_share_directory("quadruped") + \
            "/urdf/go2_description.urdf"
        self.package_dir = get_package_share_directory('quadruped')

        self.get_logger().info("Initializing IK Test Node...")

        # Initialize robot model and visualization
        self._init_robot()

        # Initialize IK solver and tasks
        self._init_ik_solver()

        # Create joint state subscriber
        self.joint_state_subscriber = self.create_subscription(
            JointState,
            '/joint_states',
            self._joint_state_callback,
            10
        )

        self.action_publisher = self.create_publisher(
            Float32MultiArray,
            '/action',
            10
        )

        # Initialize timing variables
        self.rate = RateLimiter(frequency=self.frequency)
        self.dt = self.rate.period
        self.t = 0.0  # [s]
        self.period = 2.0  # Gait period in seconds
        self.omega = 2 * np.pi / self.period
        self.swing_height = 0.15  # Foot swing height in meters
        self.swing_legs = ["FL_foot", "FR_foot", "RL_foot", "RR_foot"]

        self.action_publisher_timer = self.create_timer(
            self.dt, self._publish_action)

        self.get_logger().info("IK Test Node initialized successfully")

    def _init_robot(self):
        """Initialize robot model and visualization."""
        # root_joint = pin.JointModelFreeFlyer()
        self.robot = pin.RobotWrapper.BuildFromURDF(
            self.urdf_filename,
            package_dirs=self.package_dir,
            # root_joint=root_joint,
        )

        self.viz = start_meshcat_visualizer(self.robot)

        # # Set initial configuration
        # identity_transform = pin.SE3.Identity()
        # identity_quaternion = pin.Quaternion(identity_transform.rotation)

        # # 前七個自由度為floating base的pose, 順序為 x,y,z,qx,qy,qz,qw
        # self.q_ref = np.concatenate((
        #     np.array([0.0, 0.0, 0.0]),
        #     identity_quaternion.coeffs(),  # (x, y, z, w)
        #     np.array([0.0, 0.8, -1.57, 0.0, 0.8, -1.57,
        #              0.0, 0.8, -1.57, 0.0, 0.8, -1.57]),
        # ))
        self.q_ref = np.array(
            [0.0, 0.8, -1.57, 0.0, 0.8, -1.57, 0.0, 0.8, -1.57, 0.0, 0.8, -1.57])

        self.configuration = pink.Configuration(
            self.robot.model, self.robot.data, self.q_ref)

    def _init_ik_solver(self):
        """Initialize IK solver and tasks."""
        swing_legs = ["FL_foot", "RR_foot", "FR_foot", "RL_foot"]

        self.ik_solver = InverseKinematicsSolver(
            self.robot, ee_name_list=swing_legs, q_ref=None, rate=self.frequency
        )

        self.tasks = []
        self.foot_tasks = {}

        for foot in swing_legs:
            task = FrameTask(
                foot,
                position_cost=1.0,  # [cost] / [m]
                orientation_cost=0.0,  # [cost] / [rad]
            )
            self.tasks.append(task)
            self.foot_tasks[foot] = task

        for task in self.tasks:
            task.set_target_from_configuration(self.configuration)

        # Store initial foot positions
        self.initial_foot_positions = {
            name: task.transform_target_to_world.translation.copy()
            for name, task in self.foot_tasks.items()
        }

        # Select QP solver
        self.solver = qpsolvers.available_solvers[0]
        if "osqp" in qpsolvers.available_solvers:
            self.solver = "osqp"

    def _joint_state_callback(self, msg):
        """Callback for joint state messages."""
        self.get_logger().debug(
            f"Received joint states with {len(msg.name)} joints")
        self.current_joint_states = msg

    def _update_target_configuration_from_joint_states(self, joint_state_msg):
        """Convert ROS joint states to pin-pink configuration."""
        try:
            # Create a mapping from joint names to positions
            joint_dict = dict(
                zip(joint_state_msg.name, joint_state_msg.position))

            q_new = self.q_ref.copy()

            # Map joint states to configuration
            # This is a simplified mapping - you may need to adjust based on your robot's joint ordering
            joint_start_idx = 0  # If floating base, should skip first 7 DOFs

            for i, joint_name in enumerate(joint_state_msg.name):
                if joint_start_idx + i < len(q_new):
                    q_new[joint_start_idx + i] = joint_state_msg.position[i]

            self.target_configuration = Configuration(
                self.robot.model, self.robot.data, q_new)

        except Exception as e:
            self.get_logger().error(f"Error updating configuration: {str(e)}")

    def _compute_ik(self):
        """Compute inverse kinematics and update visualization."""
        try:
            # Update foot targets
            for leg_name in self.swing_legs:
                task = self.foot_tasks[leg_name]
                task.set_target_from_configuration(self.target_configuration)
                target_pos = task.transform_target_to_world.translation
                q = self.ik_solver.solve_ik(
                    leg_name,
                    target_pos,  # ee目標位置
                    self.configuration.q,  # 模擬傳入目前關節角度
                )
                self.configuration.update(q)  # 模擬更新目前關節角度

            # Update visualization
            self.viz.display(self.configuration.q)

            # Update time
            self.t += self.dt

        except Exception as e:
            self.get_logger().error(f"Error computing IK: {str(e)}")

    def _publish_action(self):
        """Publish a dummy action message."""
        if self.current_joint_states is None:
            self.get_logger().warning("No joint states received yet.")
            return

        # Update configuration with new joint states
        self._update_target_configuration_from_joint_states(
            self.current_joint_states)

        # Perform IK computation
        self._compute_ik()

        action_msg = Float32MultiArray()
        action_msg.data = self.configuration.q.tolist()
        self.action_publisher.publish(action_msg)
        self.get_logger().debug("Published action message.")


def main(args=None):
    """Main entry point for the ROS2 node."""
    rclpy.init(args=args)

    try:
        ik_test_node = IKTestNode()
        rclpy.spin(ik_test_node)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"Error: {e}")
    finally:
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
