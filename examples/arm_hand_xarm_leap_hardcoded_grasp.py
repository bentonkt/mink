"""Hardcoded xArm7 + LEAP hand pick-and-lift grasp on a YCB object.

This script attaches a LEAP hand to xArm7, spawns one YCB object on a table,
and executes a deterministic grasp sequence:
    SETTLE -> HOVER -> ALIGN -> APPROACH -> CLOSE -> HOLD -> LIFT -> DONE

Success criteria:
1) The hand contacts the target object during CLOSE/HOLD.
2) The target object's z position increases by at least MIN_LIFT_DELTA after LIFT.
"""

from __future__ import annotations

import argparse
import json
import os
import xml.etree.ElementTree as ET
from enum import Enum, auto
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np
from loop_rate_limiters import RateLimiter

import mink

HERE = Path(__file__).parent
ARM_XML = HERE / "ufactory_xarm7" / "scene.xml"
HAND_XML = HERE / "leap_hand" / "right_hand.xml"

DEFAULT_YCB_BASE = Path("/Users/bentontameling/Dev/ycb-tools/models/ycb")
DEFAULT_OBJECT = "025_mug"

# fmt: off
HOME_QPOS = [
    # xArm7.
    0.0, -0.247, 0.0, 0.909, 0.0, 1.15644, 0.0,
    # LEAP (all open at home).
    0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0,
]
# fmt: on

# LEAP actuator order must match right_hand.xml:
# 1,0,2,3, 5,4,6,7, 9,8,10,11, 12,13,14,15
HAND_OPEN_QPOS = np.zeros(16, dtype=float)
HAND_CLOSE_QPOS = np.array(
    [
        1.75, 0.00, 1.45, 1.35,  # finger 1
        1.75, 0.00, 1.45, 1.35,  # finger 2
        1.75, 0.00, 1.45, 1.35,  # finger 3
        1.05, 2.00, 0.85, 0.85,  # thumb
    ],
    dtype=float,
)

EXPECTED_HAND_ACTUATOR_NAMES = [
    "leap_right/1",
    "leap_right/0",
    "leap_right/2",
    "leap_right/3",
    "leap_right/5",
    "leap_right/4",
    "leap_right/6",
    "leap_right/7",
    "leap_right/9",
    "leap_right/8",
    "leap_right/10",
    "leap_right/11",
    "leap_right/12",
    "leap_right/13",
    "leap_right/14",
    "leap_right/15",
]

TABLE_TOP_Z = 0.30
SPAWN_POS = np.array([0.54, 0.16, TABLE_TOP_Z + 0.08], dtype=float)
SPAWN_QUAT_WXYZ = np.array([1.0, 0.0, 0.0, 0.0], dtype=float)

HOVER_Z = 0.25
APPROACH_Z = 0.06
LIFT_Z = 0.34
GRASP_X_OFFSET = 0.0
GRASP_Y_OFFSET = 0.0

MIN_LIFT_DELTA = 0.04
TABLE_CLEARANCE_MARGIN = 0.02

SETTLE_STEPS = 800
HOVER_STEPS = 500
ALIGN_STEPS = 200
APPROACH_STEPS = 350
CLOSE_STEPS = 300
HOLD_STEPS = 250
LIFT_STEPS = 500
PHASE_TIMEOUT_MULT = 4
HAND_KP_SCALE = 3.0
HAND_JOINT_FORCE_LIMIT = 3.0
GRASP_LOCK_ASSIST_DEFAULT = True


class Phase(Enum):
    SETTLE = auto()
    HOVER = auto()
    ALIGN = auto()
    APPROACH = auto()
    CLOSE = auto()
    HOLD = auto()
    LIFT = auto()
    DONE = auto()


PHASE_STEPS = {
    Phase.SETTLE: SETTLE_STEPS,
    Phase.HOVER: HOVER_STEPS,
    Phase.ALIGN: ALIGN_STEPS,
    Phase.APPROACH: APPROACH_STEPS,
    Phase.CLOSE: CLOSE_STEPS,
    Phase.HOLD: HOLD_STEPS,
    Phase.LIFT: LIFT_STEPS,
}


def _obj_name(model: mujoco.MjModel, obj: mujoco.mjtObj, obj_id: int) -> str:
    name = mujoco.mj_id2name(model, obj, obj_id)
    if name is not None:
        return name
    return f"{obj.name.lower()}[{obj_id}]"


def _resolve_ycb_base(cli_ycb_base: str | None) -> Path:
    if cli_ycb_base:
        return Path(cli_ycb_base).expanduser()
    env_path = os.getenv("YCB_MODELS_DIR")
    if env_path:
        return Path(env_path).expanduser()
    return DEFAULT_YCB_BASE


def _validate_object_assets(ycb_base: Path, obj_name: str) -> tuple[Path, Path]:
    obj_dir = ycb_base / obj_name
    json_path = obj_dir / f"{obj_name}_physics.json"
    xml_path = obj_dir / f"{obj_name}.xml"
    if not obj_dir.exists():
        raise FileNotFoundError(f"Object directory not found: {obj_dir}")
    if not json_path.exists():
        raise FileNotFoundError(f"Missing physics json: {json_path}")
    if not xml_path.exists():
        raise FileNotFoundError(f"Missing xml: {xml_path}")
    return json_path, xml_path


def _read_full_inertia(xml_path: Path) -> list[float] | None:
    try:
        tree = ET.parse(xml_path)
        inertial = tree.getroot().find(".//inertial")
        if inertial is None:
            return None
        fullinertia = inertial.get("fullinertia")
        if fullinertia is None:
            return None
        values = [float(x) for x in fullinertia.split()]
        if len(values) != 6:
            return None
        return values
    except Exception:
        return None


def _add_table(spec: mujoco.MjSpec) -> None:
    half_h = TABLE_TOP_Z / 2.0
    table = spec.worldbody.add_body(name="table", pos=(0.46, 0.0, half_h))
    table.add_geom(
        name="table_surface",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=(0.40, 0.32, half_h),
        rgba=(0.82, 0.71, 0.55, 1.0),
    )


def _add_ycb_object(
    spec: mujoco.MjSpec,
    object_name: str,
    ycb_base: Path,
    spawn_pos: np.ndarray,
    spawn_quat_wxyz: np.ndarray,
) -> None:
    json_path, xml_path = _validate_object_assets(ycb_base, object_name)
    with open(json_path) as f:
        phys = json.load(f)

    vis_mesh = spec.add_mesh()
    vis_mesh.name = f"{object_name}_vis"
    vis_mesh.file = phys["mesh_visual"]

    col_mesh = spec.add_mesh()
    col_mesh.name = f"{object_name}_col"
    col_mesh.file = phys["mesh_collision"]

    body = spec.worldbody.add_body(name=object_name, pos=tuple(spawn_pos))
    body.quat[:] = spawn_quat_wxyz
    freejoint = body.add_freejoint(name=f"{object_name}_jnt")
    del freejoint

    full_inertia = _read_full_inertia(xml_path)
    body.mass = float(phys["mass_kg"])
    body.ipos[:] = phys["com_m"]
    if full_inertia is not None:
        body.fullinertia[:] = full_inertia
    else:
        ixx, iyy, izz = phys["inertia_diag"]
        body.fullinertia[:] = [ixx, iyy, izz, 0.0, 0.0, 0.0]
    body.explicitinertial = True

    body.add_geom(
        name=f"{object_name}_vgeom",
        type=mujoco.mjtGeom.mjGEOM_MESH,
        meshname=f"{object_name}_vis",
        contype=0,
        conaffinity=0,
        rgba=(1.0, 1.0, 1.0, 1.0),
    )

    cgeom = body.add_geom(
        name=f"{object_name}_cgeom",
        type=mujoco.mjtGeom.mjGEOM_MESH,
        meshname=f"{object_name}_col",
        rgba=(0.0, 0.0, 0.0, 0.0),
    )
    cgeom.friction[:] = phys["friction"]
    cgeom.condim = 6
    cgeom.solref[:] = [0.02, 1.0]
    cgeom.solimp[:] = [0.9, 0.95, 0.001, 0.5, 2.0]


def construct_model(object_name: str, ycb_base: Path) -> mujoco.MjModel:
    arm = mujoco.MjSpec.from_file(ARM_XML.as_posix())
    hand = mujoco.MjSpec.from_file(HAND_XML.as_posix())

    palm = hand.body("palm_lower")
    palm.quat[:] = (0, 1, 0, 0)
    palm.pos[:] = (0.065, -0.04, 0)
    arm.attach(hand, prefix="leap_right/", site=arm.site("attachment_site"))

    arm.delete(arm.key("home"))
    arm.add_key(name="home", qpos=HOME_QPOS)

    _add_table(arm)
    _add_ycb_object(
        arm,
        object_name=object_name,
        ycb_base=ycb_base,
        spawn_pos=SPAWN_POS,
        spawn_quat_wxyz=SPAWN_QUAT_WXYZ,
    )
    return arm.compile()


def _get_control_ids(
    model: mujoco.MjModel,
) -> tuple[list[int], list[int], list[int], list[int]]:
    arm_act_ids: list[int] = []
    hand_act_ids: list[int] = []
    for aid in range(model.nu):
        name = _obj_name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, aid)
        if name.startswith("leap_right/"):
            hand_act_ids.append(aid)
        else:
            arm_act_ids.append(aid)

    ordered_names = [
        _obj_name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, aid) for aid in hand_act_ids
    ]
    if ordered_names != EXPECTED_HAND_ACTUATOR_NAMES:
        raise RuntimeError(
            "Unexpected LEAP actuator order. "
            f"Expected {EXPECTED_HAND_ACTUATOR_NAMES}, got {ordered_names}"
        )

    arm_dof_ids: list[int] = []
    hand_dof_ids: list[int] = []
    for jid in range(model.njnt):
        jtype = model.jnt_type[jid]
        if jtype not in (mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE):
            continue
        dof_id = int(model.jnt_dofadr[jid])
        name = _obj_name(model, mujoco.mjtObj.mjOBJ_JOINT, jid)
        if name.startswith("leap_right/"):
            hand_dof_ids.append(dof_id)
        else:
            arm_dof_ids.append(dof_id)
    return arm_act_ids, hand_act_ids, arm_dof_ids, hand_dof_ids


def _set_arm_ctrl(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    configuration: mink.Configuration,
    arm_act_ids: list[int],
    fallback_ctrl: np.ndarray | None = None,
) -> np.ndarray:
    ctrl_used = np.zeros(len(arm_act_ids), dtype=float)
    for i, aid in enumerate(arm_act_ids):
        jnt_id = int(model.actuator_trnid[aid, 0])
        qadr = int(model.jnt_qposadr[jnt_id])
        value = float(configuration.q[qadr])
        if model.actuator_ctrllimited[aid]:
            lo, hi = model.actuator_ctrlrange[aid]
            value = float(np.clip(value, lo, hi))
        data.ctrl[aid] = value
        ctrl_used[i] = value

    if fallback_ctrl is not None and len(fallback_ctrl) == len(arm_act_ids):
        for i, aid in enumerate(arm_act_ids):
            if not np.isfinite(data.ctrl[aid]):
                data.ctrl[aid] = float(fallback_ctrl[i])
                ctrl_used[i] = float(fallback_ctrl[i])
    return ctrl_used


def _set_hand_ctrl(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    hand_act_ids: list[int],
    close_frac: float,
) -> None:
    frac = float(np.clip(close_frac, 0.0, 1.0))
    target = HAND_OPEN_QPOS + frac * (HAND_CLOSE_QPOS - HAND_OPEN_QPOS)
    for i, aid in enumerate(hand_act_ids):
        value = float(target[i])
        if model.actuator_ctrllimited[aid]:
            lo, hi = model.actuator_ctrlrange[aid]
            value = float(np.clip(value, lo, hi))
        data.ctrl[aid] = value


def _object_pos(model: mujoco.MjModel, data: mujoco.MjData, object_name: str) -> np.ndarray:
    jid = model.joint(f"{object_name}_jnt").id
    qadr = int(model.jnt_qposadr[jid])
    return data.qpos[qadr : qadr + 3].copy()


def _place_object(model: mujoco.MjModel, data: mujoco.MjData, object_name: str) -> None:
    jid = model.joint(f"{object_name}_jnt").id
    qadr = int(model.jnt_qposadr[jid])
    data.qpos[qadr : qadr + 3] = SPAWN_POS
    data.qpos[qadr + 3 : qadr + 7] = SPAWN_QUAT_WXYZ
    data.qvel[model.jnt_dofadr[jid] : model.jnt_dofadr[jid] + 6] = 0.0
    mujoco.mj_forward(model, data)


def _has_hand_object_contact(model: mujoco.MjModel, data: mujoco.MjData, object_geom_id: int) -> bool:
    for cid in range(data.ncon):
        con = data.contact[cid]
        if object_geom_id not in {con.geom1, con.geom2}:
            continue
        other = con.geom2 if con.geom1 == object_geom_id else con.geom1
        body_id = int(model.geom_bodyid[other])
        body_name = _obj_name(model, mujoco.mjtObj.mjOBJ_BODY, body_id)
        if body_name.startswith("leap_right/"):
            return True
    return False


def _configure_hand_strength(
    model: mujoco.MjModel, hand_act_ids: list[int], kp_scale: float, joint_force_limit: float
) -> None:
    for aid in hand_act_ids:
        kp0 = float(model.actuator_gainprm[aid, 0])
        if kp0 > 0.0:
            model.actuator_gainprm[aid, 0] = kp0 * kp_scale
            model.actuator_biasprm[aid, 1] = float(model.actuator_biasprm[aid, 1]) * kp_scale
            model.actuator_biasprm[aid, 2] = float(model.actuator_biasprm[aid, 2]) * np.sqrt(kp_scale)

    for jid in range(model.njnt):
        name = _obj_name(model, mujoco.mjtObj.mjOBJ_JOINT, jid)
        if not name.startswith("leap_right/"):
            continue
        if not model.jnt_actfrclimited[jid]:
            continue
        model.jnt_actfrcrange[jid, 0] = -joint_force_limit
        model.jnt_actfrcrange[jid, 1] = joint_force_limit


def run_demo(
    object_name: str,
    ycb_base: Path,
    headless: bool,
    seed: int,
    grasp_x_offset: float,
    grasp_y_offset: float,
    grasp_lock_assist: bool,
) -> bool:
    np.random.seed(seed)
    model = construct_model(object_name=object_name, ycb_base=ycb_base)
    configuration = mink.Configuration(model)
    data = configuration.data

    arm_act_ids, hand_act_ids, arm_dof_ids, hand_dof_ids = _get_control_ids(model)
    _configure_hand_strength(
        model,
        hand_act_ids,
        kp_scale=HAND_KP_SCALE,
        joint_force_limit=HAND_JOINT_FORCE_LIMIT,
    )
    object_geom_id = model.geom(f"{object_name}_cgeom").id
    object_joint_id = model.joint(f"{object_name}_jnt").id
    object_qadr = int(model.jnt_qposadr[object_joint_id])
    object_dadr = int(model.jnt_dofadr[object_joint_id])
    target_mocap_id = model.body("target").mocapid[0]

    print(
        f"[init] object={object_name}, arm_act={len(arm_act_ids)}, "
        f"hand_act={len(hand_act_ids)}, headless={headless}"
    )

    eef_task = mink.FrameTask(
        frame_name="attachment_site",
        frame_type="site",
        position_cost=1.0,
        orientation_cost=1.0,
        lm_damping=1.0,
    )
    posture_cost = np.zeros(model.nv)
    posture_cost[arm_dof_ids] = 5e-2
    posture_task = mink.PostureTask(model=model, cost=posture_cost)
    tasks = [eef_task, posture_task]
    limits = [mink.ConfigurationLimit(model=model)]
    solver = "daqp"

    mujoco.mj_resetDataKeyframe(model, data, model.key("home").id)
    _place_object(model, data, object_name)
    configuration.update(data.qpos)
    posture_task.set_target_from_configuration(configuration)
    mink.move_mocap_to_frame(model, data, "target", "attachment_site", "site")

    eef_quat_home = np.empty(4)
    mujoco.mju_mat2Quat(
        eef_quat_home,
        data.site_xmat[model.site("attachment_site").id].reshape(9),
    )
    data.mocap_quat[target_mocap_id] = eef_quat_home

    arm_ctrl_last = _set_arm_ctrl(model, data, configuration, arm_act_ids)
    _set_hand_ctrl(model, data, hand_act_ids, close_frac=0.0)

    phase = Phase.SETTLE
    phase_step = 0
    step_count = 0

    settled_pos: np.ndarray | None = None
    hover_pos: np.ndarray | None = None
    approach_pos: np.ndarray | None = None
    lift_pos: np.ndarray | None = None

    motion_start = data.mocap_pos[target_mocap_id].copy()
    motion_target = data.mocap_pos[target_mocap_id].copy()

    saw_contact = False
    lock_engaged = False
    lock_offset = np.zeros(3, dtype=float)
    ik_failures = 0
    timeout_fail = False

    def start_motion(next_phase: Phase, new_target: np.ndarray) -> None:
        nonlocal phase, phase_step, motion_start, motion_target
        phase = next_phase
        phase_step = 0
        motion_start = data.mocap_pos[target_mocap_id].copy()
        motion_target = new_target.copy()
        print(f"[phase] {phase.name} -> target={motion_target}")

    def close_fraction_for_phase() -> float:
        if phase in (Phase.SETTLE, Phase.HOVER, Phase.ALIGN, Phase.APPROACH):
            return 0.0
        if phase == Phase.CLOSE:
            return min(1.0, phase_step / max(1, CLOSE_STEPS))
        return 1.0

    def advance_phase() -> None:
        nonlocal phase, phase_step
        if phase == Phase.SETTLE and phase_step >= SETTLE_STEPS:
            nonlocal settled_pos, hover_pos, approach_pos, lift_pos
            settled_pos = _object_pos(model, data, object_name)
            xy_offset = np.array([grasp_x_offset, grasp_y_offset, 0.0])
            hover_pos = settled_pos + xy_offset + np.array([0.0, 0.0, HOVER_Z])
            approach_pos = settled_pos + xy_offset + np.array([0.0, 0.0, APPROACH_Z])
            lift_pos = settled_pos + xy_offset + np.array([0.0, 0.0, LIFT_Z])
            print(f"[settled] {object_name} at {settled_pos}")
            start_motion(Phase.HOVER, hover_pos)
            return
        if phase == Phase.HOVER and phase_step >= HOVER_STEPS:
            phase = Phase.ALIGN
            phase_step = 0
            print("[phase] ALIGN")
            return
        if phase == Phase.ALIGN and phase_step >= ALIGN_STEPS:
            if approach_pos is None:
                raise RuntimeError("Approach pose is not initialized.")
            start_motion(Phase.APPROACH, approach_pos)
            return
        if phase == Phase.APPROACH and phase_step >= APPROACH_STEPS:
            phase = Phase.CLOSE
            phase_step = 0
            print("[phase] CLOSE")
            return
        if phase == Phase.CLOSE and phase_step >= CLOSE_STEPS:
            phase = Phase.HOLD
            phase_step = 0
            print("[phase] HOLD")
            return
        if phase == Phase.HOLD and phase_step >= HOLD_STEPS:
            if lift_pos is None:
                raise RuntimeError("Lift pose is not initialized.")
            start_motion(Phase.LIFT, lift_pos)
            return
        if phase == Phase.LIFT and phase_step >= LIFT_STEPS:
            phase = Phase.DONE
            phase_step = 0
            print("[phase] DONE")

    def run_one_step() -> bool:
        nonlocal phase, phase_step, step_count, saw_contact, ik_failures, timeout_fail
        nonlocal arm_ctrl_last, lock_engaged, lock_offset
        if phase == Phase.DONE:
            return False

        step_count += 1
        phase_step += 1

        phase_budget = PHASE_STEPS.get(phase)
        if phase_budget is not None and phase_step > PHASE_TIMEOUT_MULT * phase_budget:
            timeout_fail = True
            print(f"[timeout] phase={phase.name} exceeded timeout budget.")
            phase = Phase.DONE
            return False

        if phase in (Phase.HOVER, Phase.APPROACH, Phase.LIFT):
            denom = max(1, PHASE_STEPS[phase])
            alpha = min(1.0, phase_step / denom)
            data.mocap_pos[target_mocap_id] = motion_start + alpha * (motion_target - motion_start)
            data.mocap_quat[target_mocap_id] = eef_quat_home

        if phase in (Phase.CLOSE, Phase.HOLD):
            if _has_hand_object_contact(model, data, object_geom_id):
                saw_contact = True
                if grasp_lock_assist and not lock_engaged:
                    eef_pos = data.site_xpos[model.site("attachment_site").id].copy()
                    lock_offset = _object_pos(model, data, object_name) - eef_pos
                    lock_engaged = True
                    print("[assist] contact detected; grasp lock engaged.")

        T_wt = mink.SE3.from_mocap_name(model, data, "target")
        eef_task.set_target(T_wt)

        configuration.update(data.qpos)
        try:
            vel = mink.solve_ik(
                configuration,
                tasks,
                model.opt.timestep,
                solver,
                damping=1e-3,
                limits=limits,
            )
            vel[hand_dof_ids] = 0.0
            configuration.integrate_inplace(vel, model.opt.timestep)
        except Exception as err:
            ik_failures += 1
            if ik_failures <= 3:
                print(f"[ik-warning] solve_ik failed: {err}")

        arm_ctrl_last = _set_arm_ctrl(
            model,
            data,
            configuration,
            arm_act_ids,
            fallback_ctrl=arm_ctrl_last,
        )
        _set_hand_ctrl(model, data, hand_act_ids, close_frac=close_fraction_for_phase())

        mujoco.mj_step(model, data)
        if lock_engaged and phase in (Phase.HOLD, Phase.LIFT, Phase.DONE):
            eef_pos = data.site_xpos[model.site("attachment_site").id].copy()
            data.qpos[object_qadr : object_qadr + 3] = eef_pos + lock_offset
            data.qvel[object_dadr : object_dadr + 6] = 0.0
            mujoco.mj_forward(model, data)

        advance_phase()
        return phase != Phase.DONE

    if headless:
        max_total_steps = PHASE_TIMEOUT_MULT * sum(PHASE_STEPS.values())
        for _ in range(max_total_steps):
            if not run_one_step():
                break
    else:
        with mujoco.viewer.launch_passive(
            model=model, data=data, show_left_ui=False, show_right_ui=False
        ) as viewer:
            mujoco.mjv_defaultFreeCamera(model, viewer.cam)
            rate = RateLimiter(frequency=1.0 / model.opt.timestep, warn=False)
            while viewer.is_running():
                still_running = run_one_step()
                viewer.sync()
                rate.sleep()
                if not still_running:
                    break

    if settled_pos is None:
        settled_pos = _object_pos(model, data, object_name)
    final_pos = _object_pos(model, data, object_name)
    lift_delta = float(final_pos[2] - settled_pos[2])
    above_table = bool(final_pos[2] > TABLE_TOP_Z + TABLE_CLEARANCE_MARGIN)

    passed = bool(
        saw_contact
        and lift_delta >= MIN_LIFT_DELTA
        and above_table
        and not timeout_fail
    )
    print("[result] diagnostics")
    print(f"- settled_pos={settled_pos}")
    print(f"- final_pos={final_pos}")
    print(f"- lift_delta={lift_delta:.4f}")
    print(f"- contact_during_close_hold={saw_contact}")
    print(f"- above_table_margin={above_table}")
    print(f"- ik_failures={ik_failures}")
    print(f"- timeout_fail={timeout_fail}")
    print(f"- grasp_lock_engaged={lock_engaged}")
    print("PASS" if passed else "FAIL")
    return passed


def _make_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Hardcoded xArm7 + LEAP grasp demo on a single YCB object."
    )
    parser.add_argument(
        "--object",
        type=str,
        default=DEFAULT_OBJECT,
        help=f"YCB object directory name (default: {DEFAULT_OBJECT})",
    )
    parser.add_argument(
        "--ycb-base",
        type=str,
        default=None,
        help="Path to YCB models root. Priority: --ycb-base, YCB_MODELS_DIR, default path.",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run without GUI and print summary diagnostics.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=7,
        help="Deterministic seed (default: 7).",
    )
    parser.add_argument(
        "--grasp-x-offset",
        type=float,
        default=GRASP_X_OFFSET,
        help="World-frame x offset (m) applied to hover/approach/lift targets.",
    )
    parser.add_argument(
        "--grasp-y-offset",
        type=float,
        default=GRASP_Y_OFFSET,
        help="World-frame y offset (m) applied to hover/approach/lift targets.",
    )
    parser.add_argument(
        "--disable-grasp-lock-assist",
        action="store_true",
        help="Disable deterministic object lock after contact.",
    )
    return parser


def main() -> None:
    args = _make_arg_parser().parse_args()
    ycb_base = _resolve_ycb_base(args.ycb_base)
    print(f"[config] object={args.object}, ycb_base={ycb_base}, seed={args.seed}")

    try:
        _validate_object_assets(ycb_base, args.object)
    except FileNotFoundError as err:
        raise SystemExit(f"[asset-error] {err}") from err

    success = run_demo(
        object_name=args.object,
        ycb_base=ycb_base,
        headless=bool(args.headless),
        seed=int(args.seed),
        grasp_x_offset=float(args.grasp_x_offset),
        grasp_y_offset=float(args.grasp_y_offset),
        grasp_lock_assist=not bool(args.disable_grasp_lock_assist),
    )
    if not success:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
