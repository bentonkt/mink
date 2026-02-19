"""Hard-coded grasp sequence for the xArm7 + Leap Hand system (MuJoCo).

This script is intentionally *deterministic* and easy to tweak:
- Three objects ("YCP" placeholders) are spawned on a table at hard-coded poses.
- The xArm moves the Leap Hand palm above an object, descends, closes fingers,
  lifts the object, places it back on the table, and retreats.

Control approach (mirrors other examples in this repo):
- Use Mink IK (QP) to compute joint-space updates for the robot (arm + hand).
- Apply those joint targets through MuJoCo position-like actuators (`data.ctrl`)
  and step physics to get contacts/friction during grasp and lift.

Leap Hand joint layout (from `examples/leap_hand/right_hand.xml`):
- 3 fingers + thumb, 4 joints per digit = 16 joints total.
- Joint names are numeric in the XML and are prefixed when attached.
  For the attached right hand, joints are:
    - Finger 1: 1, 0, 2, 3
    - Finger 2: 5, 4, 6, 7
    - Finger 3: 9, 8, 10, 11
    - Thumb:   12, 13, 14, 15

The `leap_synergy_*` helpers below produce "open" and "power grasp" postures that
stay within joint limits. If your object slips, typical knobs are:
- `OBJECT_FRICTION` (increase)
- `OBJECT_MASS` (decrease)
- `GRASP_STRENGTH` (increase, capped by actuator force limits)
- `GRASP_HEIGHT_OFFSET` / `APPROACH_HEIGHT_OFFSET` (tune approach geometry)
- Table/object placement (keep within reachable workspace)
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import mujoco
import mujoco.viewer
import numpy as np

import mink

_HERE = Path(__file__).parent
_ARM_XML = _HERE / "ufactory_xarm7" / "scene.xml"
_HAND_XML = _HERE / "leap_hand" / "right_hand.xml"

# fmt: off
XARM_HOME_QPOS = [
    0.0, -0.247, 0.0, 0.909, 0.0, 1.15644, 0.0,
]
# fmt: on


# -----------------------------
# Scene parameters (easy knobs)
# -----------------------------

# Table: box centered at TABLE_POS with half-extents TABLE_SIZE.
TABLE_POS = np.array([0.55, 0.00, 0.20], dtype=np.float64)
TABLE_SIZE = np.array([0.28, 0.38, 0.20], dtype=np.float64)  # top at z=0.40
TABLE_FRICTION = (1.0, 0.005, 0.0001)

# "YCP" objects: simple cylinders.
OBJECT_RADIUS = 0.020
OBJECT_HALFHEIGHT = 0.040
OBJECT_MASS = 0.025
OBJECT_FRICTION = (1.5, 0.005, 0.0001)

# Hard-coded XY locations (on the table top), Z computed from table/object size.
YCP_XY = [
    (0.48, -0.10),
    (0.55, 0.00),
    (0.48, 0.10),
]

# Grasp geometry (relative to object center).
APPROACH_HEIGHT_OFFSET = 0.18
GRASP_HEIGHT_OFFSET = 0.07
LIFT_HEIGHT_OFFSET = 0.22

# A small XY bias helps align fingers with the object if needed.
GRASP_XY_BIAS = np.array([0.00, 0.00], dtype=np.float64)

# Hand posture "strength" (0..1) used by synergies.
GRASP_STRENGTH = 0.85

# Control loop (outer) timestep, independent from MuJoCo physics timestep.
# Keep this modest; each outer tick will run `ik_iters` QP solves and then `substeps`
# MuJoCo physics steps.
DEFAULT_CONTROL_DT = 0.02  # 50 Hz


# -----------------------------
# Leap Hand joint utilities
# -----------------------------

LEAP_PREFIX = "leap_right/"

# Joint name order for the right Leap hand model (before prefixing).
LEAP_JOINTS: tuple[str, ...] = (
    # finger 1
    "1",
    "0",
    "2",
    "3",
    # finger 2
    "5",
    "4",
    "6",
    "7",
    # finger 3
    "9",
    "8",
    "10",
    "11",
    # thumb
    "12",
    "13",
    "14",
    "15",
)


def _clamp(x: float, lo: float, hi: float) -> float:
    return float(min(max(x, lo), hi))


def leap_synergy_open() -> dict[str, float]:
    """Open hand posture (slight natural curl = more stable contacts on approach)."""
    q = {}
    for j in LEAP_JOINTS:
        q[f"{LEAP_PREFIX}{j}"] = 0.0
    # Tiny curl to avoid perfectly straight singular contacts.
    for j in ("2", "6", "10", "14"):
        q[f"{LEAP_PREFIX}{j}"] = 0.15
    return q


def leap_synergy_power_grasp(strength: float) -> dict[str, float]:
    """A simple "power grasp" synergy.

    Notes on mechanics:
    - For each finger, we increase flexion in proximal-to-distal joints.
    - Thumb is flexed and slightly opposed (in this model, opposition is achieved
      via the chain joints rather than a dedicated abduction DOF).
    """
    s = _clamp(strength, 0.0, 1.0)

    # Finger targets (within joint ranges in the XML):
    # MCP (range ~[-0.314, 2.23]), PIP (~[-1.047, 1.047]),
    # DIP (~[-0.506, 1.885]), TIP (~[-0.366, 2.042]).
    mcp = 0.6 + 1.2 * s
    pip = 0.2 + 0.8 * s
    dip = 0.4 + 1.0 * s
    tip = 0.3 + 1.1 * s

    # Thumb targets (ranges ~[-0.35..2.09], [-0.47..2.44], [-1.2..1.9], [-1.34..1.88]).
    th0 = 0.4 + 1.0 * s
    th1 = 0.6 + 1.3 * s
    th2 = 0.2 + 1.0 * s
    th3 = 0.1 + 0.8 * s

    q: dict[str, float] = {}
    # Finger 1
    q[f"{LEAP_PREFIX}1"] = mcp
    q[f"{LEAP_PREFIX}0"] = pip
    q[f"{LEAP_PREFIX}2"] = dip
    q[f"{LEAP_PREFIX}3"] = tip
    # Finger 2
    q[f"{LEAP_PREFIX}5"] = mcp
    q[f"{LEAP_PREFIX}4"] = pip
    q[f"{LEAP_PREFIX}6"] = dip
    q[f"{LEAP_PREFIX}7"] = tip
    # Finger 3
    q[f"{LEAP_PREFIX}9"] = mcp
    q[f"{LEAP_PREFIX}8"] = pip
    q[f"{LEAP_PREFIX}10"] = dip
    q[f"{LEAP_PREFIX}11"] = tip
    # Thumb
    q[f"{LEAP_PREFIX}12"] = th0
    q[f"{LEAP_PREFIX}13"] = th1
    q[f"{LEAP_PREFIX}14"] = th2
    q[f"{LEAP_PREFIX}15"] = th3
    return q


def _leap_joint_values_in_order(targets: dict[str, float]) -> list[float]:
    """Return 16 Leap joint values in `LEAP_JOINTS` order (prefixed keys)."""
    vals: list[float] = []
    for j in LEAP_JOINTS:
        vals.append(float(targets.get(f"{LEAP_PREFIX}{j}", 0.0)))
    return vals


def apply_joint_targets_to_qpos(
    model: mujoco.MjModel,
    qpos: np.ndarray,
    joint_targets: dict[str, float],
) -> np.ndarray:
    """Return a copy of `qpos` with provided hinge joint targets applied."""
    q = qpos.copy()
    for joint_name, value in joint_targets.items():
        jid = model.joint(joint_name).id
        qadr = model.jnt_qposadr[jid]
        q[qadr] = value
    return q


# -----------------------------
# MuJoCo model construction
# -----------------------------


def _table_top_z() -> float:
    return float(TABLE_POS[2] + TABLE_SIZE[2])


def construct_model() -> mujoco.MjModel:
    arm = mujoco.MjSpec.from_file(_ARM_XML.as_posix())
    hand = mujoco.MjSpec.from_file(_HAND_XML.as_posix())

    # Attach the right Leap hand at the xArm attachment site.
    # These numbers match `examples/arm_hand_xarm_leap.py` and produce a reasonable
    # palm orientation for top-down grasps.
    palm = hand.body("palm_lower")
    palm.quat[:] = (0, 1, 0, 0)
    palm.pos[:] = (0.065, -0.04, 0)
    arm.attach(hand, prefix=LEAP_PREFIX, site=arm.site("attachment_site"))

    # Add a table.
    arm.worldbody.add_geom(
        name="table",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=tuple(TABLE_SIZE.tolist()),
        pos=tuple(TABLE_POS.tolist()),
        rgba=(0.55, 0.42, 0.30, 1.0),
        contype=1,
        conaffinity=1,
        friction=TABLE_FRICTION,
    )

    # Add three "YCP" placeholder objects (free bodies).
    z0 = _table_top_z() + OBJECT_HALFHEIGHT + 1e-3
    object_qpos7: list[float] = []
    for i, (x, y) in enumerate(YCP_XY, start=1):
        body = arm.worldbody.add_body(name=f"ycp_{i}", pos=(x, y, z0))
        body.add_freejoint()
        body.add_geom(
            name=f"ycp_{i}_geom",
            type=mujoco.mjtGeom.mjGEOM_CYLINDER,
            size=(OBJECT_RADIUS, OBJECT_HALFHEIGHT),
            rgba=(0.2, 0.7, 0.5, 1.0),
            mass=OBJECT_MASS,
            friction=OBJECT_FRICTION,
            condim=6,
        )
        body.add_site(
            name=f"ycp_{i}_site",
            pos=(0.0, 0.0, 0.0),
            size=(0.004, 0.004, 0.004),
            rgba=(0.2, 1.0, 0.2, 0.7),
        )
        # Free-joint qpos ordering is (x, y, z, qw, qx, qy, qz).
        object_qpos7.extend([float(x), float(y), float(z0), 1.0, 0.0, 0.0, 0.0])

    # Replace the "home" keyframe so its qpos length matches the final model.nq.
    # This mirrors other arm+hand examples in this repo.
    try:
        home_key = arm.key("home")
        arm.delete(home_key)
    except Exception:  # noqa: BLE001
        # Some scenes may not define a home key; ignore.
        pass

    leap_open = leap_synergy_open()
    home_qpos = [*XARM_HOME_QPOS, *_leap_joint_values_in_order(leap_open), *object_qpos7]
    arm.add_key(name="home", qpos=home_qpos)

    return arm.compile()


# -----------------------------
# Control loop helpers
# -----------------------------


def _freeze_freejoint_dofs_for_bodies(
    model: mujoco.MjModel, body_names: Iterable[str]
) -> mink.DofFreezingTask:
    dof_indices: list[int] = []
    for bname in body_names:
        bid = model.body(bname).id
        jadr = model.body_jntadr[bid]
        jnum = model.body_jntnum[bid]
        for j in range(jadr, jadr + jnum):
            if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE:
                vadr = model.jnt_dofadr[j]
                dof_indices.extend(range(int(vadr), int(vadr) + 6))
    return mink.DofFreezingTask(model=model, dof_indices=dof_indices)


def _set_ctrl_from_qpos(model: mujoco.MjModel, data: mujoco.MjData) -> None:
    """Map current `data.qpos` into `data.ctrl` for joint actuators.

    This is required once the model contains unactuated free joints (objects),
    because then `len(qpos) != len(ctrl)`.
    """
    for act_id in range(model.nu):
        trntype = mujoco.mjtTrn(model.actuator_trntype[act_id])
        if trntype != mujoco.mjtTrn.mjTRN_JOINT:
            raise RuntimeError(
                f"Unsupported actuator transmission type: {trntype} for actuator {act_id}"
            )
        jnt_id = int(model.actuator_trnid[act_id, 0])
        qadr = int(model.jnt_qposadr[jnt_id])
        data.ctrl[act_id] = float(data.qpos[qadr])


@dataclass(frozen=True)
class Stage:
    name: str
    palm_pos_start: np.ndarray
    palm_pos_end: np.ndarray
    hand_targets_start: dict[str, float]
    hand_targets_end: dict[str, float]
    duration_s: float


def _lerp(a: np.ndarray, b: np.ndarray, t: float) -> np.ndarray:
    t = float(_clamp(t, 0.0, 1.0))
    return (1.0 - t) * a + t * b


def run_stage(
    *,
    model: mujoco.MjModel,
    data: mujoco.MjData,
    configuration: mink.Configuration,
    palm_task: mink.FrameTask,
    posture_task: mink.PostureTask,
    palm_rotation: mink.SO3,
    stage: Stage,
    solver: str,
    ik_iters: int,
    control_dt: float,
    substeps: int,
    viewer: mujoco.viewer.Handle | None,
    object_freeze_constraint: mink.DofFreezingTask,
) -> None:
    control_dt = float(control_dt)
    if control_dt <= 0:
        raise ValueError("control_dt must be > 0")
    if ik_iters <= 0:
        raise ValueError("ik_iters must be >= 1")
    if substeps <= 0:
        raise ValueError("substeps must be >= 1")

    dt_ik = control_dt / float(ik_iters)
    steps = max(1, int(np.ceil(stage.duration_s / control_dt)))

    for k in range(steps):
        t = k / (steps - 1) if steps > 1 else 1.0
        palm_pos = _lerp(stage.palm_pos_start, stage.palm_pos_end, t)

        # Interpolate hand posture by simple scalar blend.
        # (All Leap joints are 1-DOF hinges.)
        hand_targets: dict[str, float] = {}
        all_keys = set(stage.hand_targets_start) | set(stage.hand_targets_end)
        for key in all_keys:
            a = stage.hand_targets_start.get(key, 0.0)
            b = stage.hand_targets_end.get(key, 0.0)
            hand_targets[key] = float((1.0 - t) * a + t * b)

        target_pose = mink.SE3.from_rotation_and_translation(palm_rotation, palm_pos)
        palm_task.set_target(target_pose)

        # Update posture target to include the desired hand posture.
        q_des = apply_joint_targets_to_qpos(model, configuration.q, hand_targets)
        posture_task.set_target(q_des)

        # Multiple IK micro-iterations per physics step improves tracking.
        for _ in range(ik_iters):
            vel = mink.solve_ik(
                configuration=configuration,
                tasks=[palm_task, posture_task],
                dt=dt_ik,
                solver=solver,
                damping=1e-3,
                limits=[mink.ConfigurationLimit(model)],
                constraints=[object_freeze_constraint],
            )
            configuration.integrate_inplace(vel, dt_ik)

        # Track IK solution with actuators and step physics for contacts.
        _set_ctrl_from_qpos(model, data)
        for _ in range(substeps):
            mujoco.mj_step(model, data)
        configuration.update()  # refresh kinematics after dynamics step

        if viewer is not None:
            viewer.sync()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--object-index", type=int, default=0, choices=(0, 1, 2))
    parser.add_argument("--no-viewer", action="store_true")
    parser.add_argument("--ik-iters", type=int, default=10)
    parser.add_argument("--control-dt", type=float, default=DEFAULT_CONTROL_DT)
    parser.add_argument("--solver", type=str, default="daqp")
    args = parser.parse_args()

    model = construct_model()
    configuration = mink.Configuration(model)
    data = configuration.data

    # Initialize to home (the xArm home keyframe exists in the base scene).
    mujoco.mj_resetDataKeyframe(model, data, model.key("home").id)
    mujoco.mj_forward(model, data)
    configuration.update(data.qpos)

    # Task: control the Leap palm (more intuitive than attachment_site offsets).
    palm_task = mink.FrameTask(
        frame_name=f"{LEAP_PREFIX}palm_lower",
        frame_type="body",
        position_cost=1.0,
        orientation_cost=1.0,
        lm_damping=1.0,
    )

    # Posture regularizer: keep arm near current while letting hand converge.
    # Use a per-DOF cost vector to emphasize hand shape more than arm drift.
    posture_cost = np.zeros((model.nv,), dtype=np.float64) + 1e-4
    for j in ("joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "joint7"):
        jid = model.joint(j).id
        did = int(model.jnt_dofadr[jid])
        posture_cost[did] = 5e-4
    for j in LEAP_JOINTS:
        jid = model.joint(f"{LEAP_PREFIX}{j}").id
        did = int(model.jnt_dofadr[jid])
        posture_cost[did] = 2e-2
    posture_task = mink.PostureTask(model=model, cost=posture_cost)
    posture_task.set_target_from_configuration(configuration)

    # Fix object DOFs during IK solve (but not during physics): objects should move only
    # due to contact dynamics, not IK numerical drift.
    object_names = [f"ycp_{i}" for i in (1, 2, 3)]
    object_freeze_constraint = _freeze_freejoint_dofs_for_bodies(model, object_names)

    # Use the home palm orientation as a constant "top-down" grasp orientation.
    T_palm_home = configuration.get_transform_frame_to_world(
        f"{LEAP_PREFIX}palm_lower", "body"
    )
    palm_rotation = T_palm_home.rotation()

    # Choose target object.
    obj_name = object_names[args.object_index]
    obj_id = model.body(obj_name).id
    obj_pos0 = data.xpos[obj_id].copy()

    # Compute stage waypoints.
    grasp_xy = obj_pos0[:2] + GRASP_XY_BIAS
    grasp_center = np.array([grasp_xy[0], grasp_xy[1], obj_pos0[2]], dtype=np.float64)
    pos_approach = grasp_center + np.array([0.0, 0.0, APPROACH_HEIGHT_OFFSET])
    pos_grasp = grasp_center + np.array([0.0, 0.0, GRASP_HEIGHT_OFFSET])
    pos_lift = grasp_center + np.array([0.0, 0.0, LIFT_HEIGHT_OFFSET])

    hand_open = leap_synergy_open()
    hand_closed = leap_synergy_power_grasp(GRASP_STRENGTH)

    stages = [
        Stage(
            name="approach_above_object",
            palm_pos_start=T_palm_home.translation().copy(),
            palm_pos_end=pos_approach,
            hand_targets_start=hand_open,
            hand_targets_end=hand_open,
            duration_s=2.0,
        ),
        Stage(
            name="descend_to_grasp_height",
            palm_pos_start=pos_approach,
            palm_pos_end=pos_grasp,
            hand_targets_start=hand_open,
            hand_targets_end=hand_open,
            duration_s=2.0,
        ),
        Stage(
            name="close_fingers",
            palm_pos_start=pos_grasp,
            palm_pos_end=pos_grasp,
            hand_targets_start=hand_open,
            hand_targets_end=hand_closed,
            duration_s=2.0,
        ),
        Stage(
            name="lift",
            palm_pos_start=pos_grasp,
            palm_pos_end=pos_lift,
            hand_targets_start=hand_closed,
            hand_targets_end=hand_closed,
            duration_s=2.0,
        ),
        Stage(
            name="lower_to_table",
            palm_pos_start=pos_lift,
            palm_pos_end=pos_grasp,
            hand_targets_start=hand_closed,
            hand_targets_end=hand_closed,
            duration_s=2.0,
        ),
        Stage(
            name="open_to_release",
            palm_pos_start=pos_grasp,
            palm_pos_end=pos_grasp,
            hand_targets_start=hand_closed,
            hand_targets_end=hand_open,
            duration_s=1.5,
        ),
        Stage(
            name="retreat",
            palm_pos_start=pos_grasp,
            palm_pos_end=pos_approach,
            hand_targets_start=hand_open,
            hand_targets_end=hand_open,
            duration_s=2.0,
        ),
    ]

    # Optional viewer (best-effort; headless runs should use --no-viewer).
    viewer = None
    if not args.no_viewer:
        try:
            viewer = mujoco.viewer.launch_passive(
                model=model, data=data, show_left_ui=False, show_right_ui=False
            )
        except Exception as exc:  # noqa: BLE001
            print(f"[warn] Viewer unavailable ({exc}); continuing headless.")
            viewer = None

    print(f"Selected object: {obj_name} at {obj_pos0}")
    print(f"Table top z: {_table_top_z():.3f}")
    physics_dt = float(model.opt.timestep)
    substeps = max(1, int(round(float(args.control_dt) / physics_dt)))
    control_dt = substeps * physics_dt
    print(f"Physics dt: {physics_dt:.4f}s | Control dt: {control_dt:.4f}s | substeps: {substeps}")

    # Run the scripted sequence.
    for stage in stages:
        print(f"Stage: {stage.name}")
        run_stage(
            model=model,
            data=data,
            configuration=configuration,
            palm_task=palm_task,
            posture_task=posture_task,
            palm_rotation=palm_rotation,
            stage=stage,
            solver=args.solver,
            ik_iters=args.ik_iters,
            control_dt=control_dt,
            substeps=substeps,
            viewer=viewer,
            object_freeze_constraint=object_freeze_constraint,
        )

    # Basic sanity checks (headless-friendly):
    obj_pos_end = data.xpos[obj_id].copy()
    dz = float(obj_pos_end[2] - obj_pos0[2])
    print(f"Object Δz after sequence: {dz:.4f} m (end - start)")

    if viewer is not None:
        # Keep the viewer open; maintain current control targets.
        print("Sequence complete. Viewer is still running; close the window to exit.")
        while viewer.is_running():
            _set_ctrl_from_qpos(model, data)
            mujoco.mj_step(model, data)
            viewer.sync()


if __name__ == "__main__":
    main()

