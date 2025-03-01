"""Example demonstrating the use of ConstraintsTask with Agility Cassie robot."""

from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np
from loop_rate_limiters import RateLimiter
import time
import mink
from mink.tasks import ConstraintsTask
from mink.lie import SE3

_HERE = Path(__file__).parent
_XML = _HERE / "agility_cassie" / "scene.xml"


def main():
    """Run the constraints task example with Agility Cassie robot."""
    # Load the Cassie model
    model = mujoco.MjModel.from_xml_path(_XML.as_posix())
    configuration = mink.Configuration(model)

    # Define the feet for visualization and control
    feet = ["left-foot", "right-foot"]

    # Create tasks
    tasks = [
        pelvis_orientation_task := mink.FrameTask(
            frame_name="cassie-pelvis",
            frame_type="body",
            position_cost=0.0,
            orientation_cost=10.0,
        ),
        posture_task := mink.PostureTask(model, cost=1.0),
        com_task := mink.ComTask(cost=200.0),
        # Add the constraints task to enforce the equality constraints in the model
        constraints_task := ConstraintsTask(
            constraint_cost=500.0,  # High cost to prioritize constraint satisfactions
            gain=0.5,               # Moderate gain for stability
            lm_damping=1e-3,        # Small damping for numerical stability
        ),
    ]

    # Add feet tasks
    feet_tasks = []
    for foot in feet:
        task = mink.FrameTask(
            frame_name=foot,
            frame_type="body",  # Cassie uses body for feet, not sites
            position_cost=200.0,
            orientation_cost=10.0,
            lm_damping=1.0,
        )
        feet_tasks.append(task)
    tasks.extend(feet_tasks)

    com_mid = model.body("com_target").mocapid[0]
    feet_mid = [model.body(f"{foot}_target").mocapid[0] for foot in feet]

    model = configuration.model
    data = configuration.data
    solver = "quadprog"

    with mujoco.viewer.launch_passive(
        model=model, data=data, show_left_ui=False, show_right_ui=False
    ) as viewer:
        mujoco.mjv_defaultFreeCamera(model, viewer.cam)
        configuration.update_from_keyframe("home")
        mujoco.mj_forward(model, data)
        posture_task.set_target_from_configuration(configuration)
        pelvis_orientation_task.set_target_from_configuration(configuration)
        for i, foot in enumerate(feet):
            mink.move_mocap_to_frame(model, data, f"{foot}_target", foot, "body")
        data.mocap_pos[com_mid] = data.subtree_com[1]
        rate = RateLimiter(frequency=200.0, warn=False)
        while viewer.is_running():
            com_task.set_target(data.mocap_pos[com_mid])
            for i, foot_task in enumerate(feet_tasks):
                foot_task.set_target(SE3.from_mocap_id(data, feet_mid[i]))
            vel = mink.solve_ik(configuration, tasks, rate.dt, solver, 1e-1)
            configuration.integrate_inplace(vel, rate.dt)
            mujoco.mj_camlight(model, data)
            viewer.sync()
            rate.sleep()


if __name__ == "__main__":
    main()