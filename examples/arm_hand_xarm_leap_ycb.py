"""xArm7 + LEAP hand grasping YCB objects.

Loads three YCB objects onto a table and executes a hardcoded pick-and-place
sequence on the mug using the LEAP hand.

Physics are enabled (mj_step) so objects settle under gravity and interact
with the hand via contact.

Usage:
    # macOS
    ./fix_mjpython_macos.sh
    uv run mjpython examples/arm_hand_xarm_leap_ycb.py

    # Linux
    uv run python examples/arm_hand_xarm_leap_ycb.py
"""

from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from enum import Enum, auto
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np
from loop_rate_limiters import RateLimiter

import mink

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_HERE = Path(__file__).parent
_ARM_XML = _HERE / "ufactory_xarm7" / "scene.xml"
_HAND_XML = _HERE / "leap_hand" / "right_hand.xml"
_YCB_BASE = Path("/Users/bentontameling/Dev/ycb-tools/models/ycb")

# ---------------------------------------------------------------------------
# Robot home posture (xArm7 + LEAP hand, all fingers open)
# ---------------------------------------------------------------------------
# fmt: off
HOME_QPOS = [
    # xArm7 joints (7 DOF)
    0.0, -0.247, 0.0, 0.909, 0.0, 1.15644, 0.0,
    # LEAP hand joints (16 DOF, all zero = open/neutral)
    0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0,
]
# fmt: on

# ---------------------------------------------------------------------------
# Hand open/close target joint positions.
# Order matches the XML actuator list: joints 1,0,2,3, 5,4,6,7, 9,8,10,11, 12,13,14,15.
# Indices 1, 5, 9 are MCP abduction joints (range ±1.047) — kept at 0 so fingers
# don't splay sideways during the grasp.
# ---------------------------------------------------------------------------
HAND_OPEN_QPOS = [0.0] * 16  # all zeros — natural flat hand

# fmt: off
HAND_CLOSE_QPOS = [
    1.5, 0.0, 1.2, 1.0,   # finger 1: MCP flex, MCP ab=0, PIP, DIP
    1.5, 0.0, 1.2, 1.0,   # finger 2: MCP flex, MCP ab=0, PIP, DIP
    1.5, 0.0, 1.2, 1.0,   # finger 3: MCP flex, MCP ab=0, PIP, DIP
    0.8, 1.5, 0.5, 0.5,   # thumb: joints 12, 13, 14, 15
]
# fmt: on

# ---------------------------------------------------------------------------
# YCB objects to place on the table.
# Each entry: (object_id, spawn_xyz, spawn_quat_wxyz, col_override_dict | None)
# Objects are spawned above the table and settle under gravity.
# ---------------------------------------------------------------------------
TABLE_TOP_Z = 0.30  # table surface height (m)

# ---------------------------------------------------------------------------
# Grasp waypoint Z offsets (relative to the settled target body origin).
# XY is filled in dynamically after SETTLE by reading the object's actual position.
# ---------------------------------------------------------------------------
HOVER_Z_OFFSET    = 0.25   # m above object COM at hover/align height
APPROACH_Z_OFFSET = 0.12   # m above object body origin at grasp height
LIFT_Z_OFFSET     = 0.32   # m above object COM after lifting

PLACE_POS    = np.array([0.30, 0.22, 0.50])
RETRACT_POS  = np.array([0.30, 0.22, 0.65])

_Q_IDENTITY = [1.0, 0.0, 0.0, 0.0]

YCB_OBJECTS: list[tuple[str, list[float], list[float], dict | None]] = [
    # Mustard bottle kept as a stable obstacle away from the mug grasp corridor.
    ("006_mustard_bottle", [0.32, -0.22, TABLE_TOP_Z + 0.02], _Q_IDENTITY, None),
    # Gelatin box: small cardboard box, settles stably flat on table.
    ("009_gelatin_box",    [0.38, -0.18, TABLE_TOP_Z + 0.08], _Q_IDENTITY, None),
    # Mug: z-up, stable resting.
    ("025_mug",            [0.54,  0.16, TABLE_TOP_Z + 0.08], _Q_IDENTITY, None),
]

# The object whose freejoint is read to compute grasp waypoints
GRASP_TARGET = "025_mug"

# ---------------------------------------------------------------------------
# Grasp state machine
# ---------------------------------------------------------------------------
class Phase(Enum):
    SETTLE   = auto()   # Let objects fall and rest on the table
    HOVER    = auto()   # Move EEF to above the target object
    ALIGN    = auto()   # Hold directly above settled object before descending
    APPROACH = auto()   # Lower EEF to grasp height
    CLOSE    = auto()   # Close fingers around object
    LIFT     = auto()   # Lift arm (object should come with it)
    PLACE    = auto()   # Transport to place position
    OPEN     = auto()   # Open fingers, release object
    RETRACT  = auto()   # Lift arm clear
    DONE     = auto()   # Hold final pose


# How many sim steps to hold each phase before advancing (200 Hz → 1 s = 200 steps)
SETTLE_STEPS  = 800   # ~4.0 s — let objects fall and come to rest
HOVER_STEPS   = 600   # ~3.0 s — arm travels from home to hover height
ALIGN_STEPS   = 200   # ~1.0 s — hold above settled object, confirm centering
APPROACH_STEPS = 400  # ~2.0 s — lower into grasp position
CLOSE_STEPS   = 300   # ~1.5 s — close fingers around object

GRASP_FRAC = 1.0   # hand_frac target for a closed grip
LIFT_STEPS    = 400   # ~2.0 s — lift object clear of table
PLACE_STEPS   = 600   # ~3.0 s — transport to place position
OPEN_STEPS    = 200   # ~1.0 s — release object
RETRACT_STEPS = 400   # ~2.0 s — lift arm clear


# ---------------------------------------------------------------------------
# Build the combined MuJoCo model
# ---------------------------------------------------------------------------

def _read_full_inertia(xml_path: Path) -> list[float] | None:
    """Parse the <inertial fullinertia="..."> attribute from a ycb-tools XML.

    Returns [Ixx, Iyy, Izz, Ixy, Ixz, Iyz] if present, else None.
    The XML uses <mujocoinclude> as root so we search for the first <inertial>
    element anywhere in the tree.
    """
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        inertial = root.find(".//inertial")
        if inertial is not None:
            fi = inertial.get("fullinertia")
            if fi:
                return [float(v) for v in fi.split()]
    except Exception:
        pass
    return None


def _add_ycb_object(
    spec: mujoco.MjSpec,
    obj_name: str,
    spawn_pos: list[float],
    ycb_base: Path,
    spawn_quat: list[float] | None = None,
    col_box_halfextents: list[float] | None = None,
    col_box_pos: list[float] | None = None,
) -> None:
    """Add one YCB object (meshes + body + freejoint + inertia) to *spec*.

    Args:
        spawn_quat: Initial orientation as [w, x, y, z]. Defaults to identity.
        col_box_halfextents: If provided, replace the convex-hull collision mesh
            with a box primitive of these half-extents [hx, hy, hz].  Use this
            for box-shaped objects: scan-mesh convex hulls have slightly non-flat
            faces (scan noise) that let the box balance on a corner.  A primitive
            has perfectly flat faces and eliminates this problem permanently.
        col_box_pos: Position offset of the box primitive in the body frame.
            Defaults to [0, 0, 0] (body origin).
    """
    json_path = ycb_base / obj_name / f"{obj_name}_physics.json"
    xml_path  = ycb_base / obj_name / f"{obj_name}.xml"
    with open(json_path) as f:
        phys = json.load(f)

    # --- mesh assets --------------------------------------------------------
    vis_mesh = spec.add_mesh()
    vis_mesh.name = f"{obj_name}_vis"
    vis_mesh.file = phys["mesh_visual"]

    # Only load the collision hull if we're not overriding with a primitive
    if col_box_halfextents is None:
        col_mesh = spec.add_mesh()
        col_mesh.name = f"{obj_name}_col"
        col_mesh.file = phys["mesh_collision"]

    # --- free body ----------------------------------------------------------
    body = spec.worldbody.add_body()
    body.name = obj_name
    body.pos[:] = spawn_pos

    # Freejoint so the object can move under gravity / contact
    jnt = body.add_freejoint()
    jnt.name = f"{obj_name}_jnt"

    # --- inertia ------------------------------------------------------------
    # Use the full 6-component tensor from the XML so off-diagonal terms
    # (which couple rotational axes) are physically correct.
    full_inertia = _read_full_inertia(xml_path)
    body.mass = phys["mass_kg"]
    body.ipos[:] = phys["com_m"]
    if full_inertia is not None:
        body.fullinertia[:] = full_inertia
    else:
        ixx, iyy, izz = phys["inertia_diag"]
        body.fullinertia[:] = [ixx, iyy, izz, 0.0, 0.0, 0.0]
    body.explicitinertial = True

    # --- visual geom (no collision) ----------------------------------------
    vgeom = body.add_geom()
    vgeom.name = f"{obj_name}_vgeom"
    vgeom.type = mujoco.mjtGeom.mjGEOM_MESH
    vgeom.meshname = f"{obj_name}_vis"
    vgeom.contype = 0
    vgeom.conaffinity = 0
    vgeom.rgba[:] = [1.0, 1.0, 1.0, 1.0]

    # --- collision geom -----------------------------------------------------
    cgeom = body.add_geom()
    cgeom.name = f"{obj_name}_cgeom"
    cgeom.friction[:] = phys["friction"]
    cgeom.condim = 6
    cgeom.rgba[:] = [0.0, 0.0, 0.0, 0.0]   # invisible
    cgeom.solref[:] = [0.02, 1.0]
    cgeom.solimp[:] = [0.9, 0.95, 0.001, 0.5, 2.0]

    if col_box_halfextents is not None:
        # --- Box primitive collision (preferred for box-shaped objects) -----
        # Scan-mesh convex hulls of rectangular objects have slightly curved
        # "flat" faces due to 3D-scan noise.  This lets the box find a
        # false equilibrium balanced on a corner edge — impossible to fix
        # with contact parameters alone.  A box primitive has mathematically
        # flat faces and correct contact normals, eliminating the issue.
        cgeom.type = mujoco.mjtGeom.mjGEOM_BOX
        cgeom.size[:3] = col_box_halfextents
        if col_box_pos is not None:
            cgeom.pos[:] = col_box_pos
    else:
        # --- Mesh convex-hull collision (default for curved objects) --------
        cgeom.type = mujoco.mjtGeom.mjGEOM_MESH
        cgeom.meshname = f"{obj_name}_col"


def _add_table(spec: mujoco.MjSpec) -> None:
    """Add a solid wooden table to the scene.

    The table top surface sits at z = TABLE_TOP_Z.
    """
    half_h = TABLE_TOP_Z / 2.0
    table = spec.worldbody.add_body()
    table.name = "table"
    table.pos[:] = [0.46, 0.0, half_h]

    geom = table.add_geom()
    geom.name = "table_surface"
    geom.type = mujoco.mjtGeom.mjGEOM_BOX
    geom.size[:3] = [0.40, 0.32, half_h]
    geom.rgba[:] = [0.82, 0.71, 0.55, 1.0]
    # Table is static — no freejoint, default contype=1/conaffinity=1
    # (defaults inherited; no need to set explicitly)


def construct_model() -> mujoco.MjModel:
    """Assemble the full model: xArm7 + LEAP hand + table + YCB objects."""
    arm = mujoco.MjSpec.from_file(_ARM_XML.as_posix())
    hand = mujoco.MjSpec.from_file(_HAND_XML.as_posix())

    # Attach LEAP hand to the xArm7 wrist (same as arm_hand_xarm_leap.py)
    palm = hand.body("palm_lower")
    palm.quat[:] = (0, 1, 0, 0)
    palm.pos[:] = (0.065, -0.04, 0)
    arm.attach(hand, prefix="leap_right/", site=arm.site("attachment_site"))

    # Replace home keyframe to include all 23 DOF
    arm.delete(arm.key("home"))
    arm.add_key(name="home", qpos=HOME_QPOS)

    # Add table
    _add_table(arm)

    # Add YCB objects
    for obj_name, spawn_pos, spawn_quat, col_override in YCB_OBJECTS:
        _add_ycb_object(
            arm, obj_name, spawn_pos, _YCB_BASE, spawn_quat,
            col_box_halfextents=(col_override or {}).get("col_box_halfextents"),
            col_box_pos=(col_override or {}).get("col_box_pos"),
        )

    # Add finger mocap targets (visual aids for hand orientation)
    fingers = ["tip_1", "tip_2", "tip_3", "th_tip"]
    for finger in fingers:
        body = arm.worldbody.add_body(name=f"{finger}_target", mocap=True)
        body.add_geom(
            type=mujoco.mjtGeom.mjGEOM_SPHERE,
            size=(0.015,) * 3,
            contype=0,
            conaffinity=0,
            rgba=(0.5, 0.2, 0.2, 0.3),
        )

    return arm.compile()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_control_ids(
    model: mujoco.MjModel,
) -> tuple[list[int], list[int], list[int], list[int]]:
    """Return (arm_actuator_ids, hand_actuator_ids, arm_dof_ids, hand_dof_ids)."""
    arm_act_ids, hand_act_ids = [], []
    for i in range(model.nu):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i) or ""
        if name.startswith("leap_right/"):
            hand_act_ids.append(i)
        else:
            arm_act_ids.append(i)

    arm_dof_ids, hand_dof_ids = [], []
    for j in range(model.njnt):
        jnt_type = model.jnt_type[j]
        if jnt_type not in (mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE):
            continue
        dof_id = int(model.jnt_dofadr[j])
        jnt_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j) or ""
        if jnt_name.startswith("leap_right/"):
            hand_dof_ids.append(dof_id)
        else:
            arm_dof_ids.append(dof_id)
    return arm_act_ids, hand_act_ids, arm_dof_ids, hand_dof_ids


def _set_arm_ctrl(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    arm_act_ids: list[int],
    configuration: mink.Configuration,
) -> None:
    """Set each arm actuator ctrl to match the IK solution (position servo)."""
    for aid in arm_act_ids:
        jnt_id = int(model.actuator_trnid[aid, 0])
        qpos_adr = int(model.jnt_qposadr[jnt_id])
        data.ctrl[aid] = configuration.q[qpos_adr]


def _set_hand_ctrl(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    hand_act_ids: list[int],
    open_frac: float,
) -> None:
    """Set hand actuator controls by interpolating between explicit target poses.

    open_frac = 0.0 → HAND_OPEN_QPOS (all zeros, flat hand)
    open_frac = 1.0 → HAND_CLOSE_QPOS (curled flexion joints, abduction joints neutral)
    """
    for i, aid in enumerate(hand_act_ids):
        data.ctrl[aid] = (
            HAND_OPEN_QPOS[i] + open_frac * (HAND_CLOSE_QPOS[i] - HAND_OPEN_QPOS[i])
        )


def _eef_pos(model: mujoco.MjModel, data: mujoco.MjData) -> np.ndarray:
    """Current world position of the attachment_site (EEF)."""
    return data.site_xpos[model.site("attachment_site").id].copy()


def _obj_pos(model: mujoco.MjModel, data: mujoco.MjData, obj_name: str) -> np.ndarray:
    """Current world position of a free body's origin (first 3 values of freejoint qpos)."""
    jnt_id = model.joint(f"{obj_name}_jnt").id
    adr = int(model.jnt_qposadr[jnt_id])
    return data.qpos[adr : adr + 3].copy()



# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _place_objects(model: mujoco.MjModel, data: mujoco.MjData) -> None:
    """Set freejoint qpos for each YCB object to its spawn position/orientation.

    Must be called AFTER mj_resetDataKeyframe (which only fills arm+hand qpos)
    because body.pos on a freejoint body does not seed qpos0.
    """
    for obj_name, spawn_pos, spawn_quat, _col_override in YCB_OBJECTS:
        jnt_id = model.joint(f"{obj_name}_jnt").id
        adr = int(model.jnt_qposadr[jnt_id])
        data.qpos[adr : adr + 3] = spawn_pos
        data.qpos[adr + 3 : adr + 7] = spawn_quat  # [w, x, y, z]
    mujoco.mj_forward(model, data)


def main() -> None:  # noqa: C901
    model = construct_model()
    arm_act_ids, hand_act_ids, arm_dof_ids, hand_dof_ids = _get_control_ids(model)
    print(f"[init] arm actuators: {len(arm_act_ids)}, hand actuators: {len(hand_act_ids)}")

    configuration = mink.Configuration(model)

    # IK tasks: arm end-effector position/orientation + posture regularisation.
    # Fingers are controlled directly via actuator ctrl (not IK tasks) so that
    # the hand open/close loop is decoupled from the arm IK.
    eef_task = mink.FrameTask(
        frame_name="attachment_site",
        frame_type="site",
        position_cost=1.0,
        orientation_cost=1.0,
        lm_damping=1.0,
    )
    # Keep posture regularisation on arm DOFs only, not the hand DOFs.
    posture_cost = np.zeros(model.nv)
    posture_cost[arm_dof_ids] = 5e-2
    posture_task = mink.PostureTask(model=model, cost=posture_cost)
    tasks = [eef_task, posture_task]
    limits = [mink.ConfigurationLimit(model=model)]
    solver = "daqp"

    data = configuration.data

    with mujoco.viewer.launch_passive(
        model=model, data=data, show_left_ui=False, show_right_ui=False
    ) as viewer:
        mujoco.mjv_defaultFreeCamera(model, viewer.cam)

        # Reset to home; initialise posture target
        mujoco.mj_resetDataKeyframe(model, data, model.key("home").id)
        # Place YCB objects at their spawn positions (must be done after keyframe reset)
        _place_objects(model, data)
        configuration.update(data.qpos)
        posture_task.set_target_from_configuration(configuration)

        # Initialise arm ctrl to home (so physics starts in a stable pose)
        _set_arm_ctrl(model, data, arm_act_ids, configuration)
        _set_hand_ctrl(model, data, hand_act_ids, open_frac=0.0)

        # Snap EEF + finger mocap targets to current robot pose
        mink.move_mocap_to_frame(model, data, "target", "attachment_site", "site")
        for finger in ["tip_1", "tip_2", "tip_3", "th_tip"]:
            mink.move_mocap_to_frame(
                model, data, f"{finger}_target", f"leap_right/{finger}", "site"
            )

        # Capture initial EEF orientation quaternion — kept fixed through grasps
        # so the hand always points in the same direction as at home.
        eef_quat_home = np.empty(4)
        mujoco.mju_mat2Quat(
            eef_quat_home,
            data.site_xmat[model.site("attachment_site").id].reshape(9),
        )

        # ---------------------------------------------------------------
        # State machine
        # ---------------------------------------------------------------
        phase = Phase.SETTLE
        step_count = 0

        # Dynamically computed waypoints (filled in after SETTLE by reading
        # the settled object position via _obj_pos).
        dyn_hover_pos:    np.ndarray | None = None
        dyn_approach_pos: np.ndarray | None = None
        dyn_lift_pos:     np.ndarray | None = None

        # Linear interpolation state for arm motion phases.
        # The mocap target moves smoothly from motion_start → motion_target
        # over the duration of the current phase.
        mocap_id = model.body("target").mocapid[0]
        motion_start  = data.mocap_pos[mocap_id].copy()
        motion_target = data.mocap_pos[mocap_id].copy()

        # Steps allocated to each motion phase (same as the *_STEPS constants).
        _MOTION_STEPS = {
            Phase.HOVER:    HOVER_STEPS,
            Phase.APPROACH: APPROACH_STEPS,
            Phase.LIFT:     LIFT_STEPS,
            Phase.PLACE:    PLACE_STEPS,
            Phase.RETRACT:  RETRACT_STEPS,
        }

        def _start_motion(target: np.ndarray, next_phase: Phase, label: str) -> None:
            """Transition to a motion phase, recording the interpolation start."""
            nonlocal phase, step_count, motion_start, motion_target
            nonlocal dyn_hover_pos, dyn_approach_pos, dyn_lift_pos
            motion_start  = data.mocap_pos[mocap_id].copy()
            motion_target = target.copy()
            phase = next_phase
            step_count = 0
            print(f"[→ {label}] {motion_start} → {motion_target}")

        rate = RateLimiter(frequency=200.0, warn=False)

        while viewer.is_running():
            configuration.update(data.qpos)

            # ---- Phase logic -------------------------------------------
            if phase == Phase.SETTLE:
                step_count += 1
                if step_count >= SETTLE_STEPS:
                    obj_p = _obj_pos(model, data, GRASP_TARGET)
                    print(f"[SETTLE done] {GRASP_TARGET} settled at {obj_p}")
                    dyn_hover_pos    = np.array([obj_p[0], obj_p[1], obj_p[2] + HOVER_Z_OFFSET])
                    dyn_approach_pos = np.array([obj_p[0], obj_p[1], obj_p[2] + APPROACH_Z_OFFSET])
                    dyn_lift_pos     = np.array([obj_p[0], obj_p[1], obj_p[2] + LIFT_Z_OFFSET])
                    _start_motion(dyn_hover_pos, Phase.HOVER, "HOVER")

            elif phase == Phase.HOVER:
                step_count += 1
                if step_count >= HOVER_STEPS:
                    phase = Phase.ALIGN
                    step_count = 0
                    print(f"[→ ALIGN] holding above {GRASP_TARGET} center…")

            elif phase == Phase.ALIGN:
                step_count += 1
                if step_count >= ALIGN_STEPS:
                    _start_motion(dyn_approach_pos, Phase.APPROACH, "APPROACH")

            elif phase == Phase.APPROACH:
                step_count += 1
                if step_count >= APPROACH_STEPS:
                    phase = Phase.CLOSE
                    step_count = 0
                    print("[→ CLOSE] closing fingers…")

            elif phase == Phase.CLOSE:
                step_count += 1
                if step_count >= CLOSE_STEPS:
                    _start_motion(dyn_lift_pos, Phase.LIFT, "LIFT")

            elif phase == Phase.LIFT:
                step_count += 1
                if step_count >= LIFT_STEPS:
                    _start_motion(PLACE_POS, Phase.PLACE, "PLACE")

            elif phase == Phase.PLACE:
                step_count += 1
                if step_count >= PLACE_STEPS:
                    phase = Phase.OPEN
                    step_count = 0
                    print("[→ OPEN] releasing object…")

            elif phase == Phase.OPEN:
                step_count += 1
                if step_count >= OPEN_STEPS:
                    _start_motion(RETRACT_POS, Phase.RETRACT, "RETRACT")

            elif phase == Phase.RETRACT:
                step_count += 1
                if step_count >= RETRACT_STEPS:
                    phase = Phase.DONE
                    print("[→ DONE] grasp sequence complete.")

            # ---- Interpolate mocap target for motion phases -------------
            if phase in _MOTION_STEPS:
                alpha = min(1.0, step_count / _MOTION_STEPS[phase])
                data.mocap_pos[mocap_id] = (
                    motion_start + alpha * (motion_target - motion_start)
                )

            # ---- Hand open / close -------------------------------------
            if phase in (Phase.SETTLE, Phase.HOVER, Phase.ALIGN, Phase.APPROACH,
                         Phase.RETRACT, Phase.DONE):
                hand_frac = 0.0
            elif phase in (Phase.LIFT, Phase.PLACE):
                hand_frac = GRASP_FRAC   # keep grip closed while transporting
            elif phase == Phase.CLOSE:
                # Ramp from 0 → GRASP_FRAC over CLOSE_STEPS
                hand_frac = min(GRASP_FRAC, GRASP_FRAC * step_count / CLOSE_STEPS)
            elif phase == Phase.OPEN:
                # Ramp from GRASP_FRAC → 0 over OPEN_STEPS
                hand_frac = max(0.0, GRASP_FRAC * (1.0 - step_count / OPEN_STEPS))
            else:
                hand_frac = 0.0

            # ---- IK step -----------------------------------------------
            # EEF task target follows the mocap body "target"
            T_wt = mink.SE3.from_mocap_name(model, data, "target")
            eef_task.set_target(T_wt)

            vel = mink.solve_ik(
                configuration, tasks, rate.dt, solver, damping=1e-3, limits=limits
            )
            # Fingers are commanded directly by actuators, so do not integrate
            # hand DOFs from IK.
            vel[hand_dof_ids] = 0.0
            configuration.integrate_inplace(vel, rate.dt)

            # ---- Actuator control + physics step -----------------------
            _set_arm_ctrl(model, data, arm_act_ids, configuration)
            _set_hand_ctrl(model, data, hand_act_ids, open_frac=hand_frac)
            mujoco.mj_step(model, data)

            # ---- Sync finger tip mocap spheres to actual finger sites ----
            for finger in ["tip_1", "tip_2", "tip_3", "th_tip"]:
                mink.move_mocap_to_frame(
                    model, data, f"{finger}_target", f"leap_right/{finger}", "site"
                )

            # ---- Viewer sync -------------------------------------------
            viewer.sync()
            rate.sleep()


if __name__ == "__main__":
    main()
