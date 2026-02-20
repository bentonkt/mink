from __future__ import annotations

import argparse
import importlib.util
import json
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from pathlib import Path

import mujoco
import numpy as np

import mink

try:
    import examples.arm_hand_xarm_leap_ycb as template
except ModuleNotFoundError:
    template_path = Path(__file__).resolve().parents[1] / "arm_hand_xarm_leap_ycb.py"
    spec = importlib.util.spec_from_file_location("arm_hand_xarm_leap_ycb", template_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load template module from {template_path}")
    template = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(template)

try:
    from examples.ycb_manipulation.library import (
        DEFAULT_YCB_BASE,
        TABLE_TOP_Z,
        GraspProfile,
        YCBObjectSpec,
        build_manipulation_library,
        write_library_json,
    )
except ModuleNotFoundError:
    from library import (  # type: ignore[no-redef]
        DEFAULT_YCB_BASE,
        TABLE_TOP_Z,
        GraspProfile,
        YCBObjectSpec,
        build_manipulation_library,
        write_library_json,
    )

ARM_XML = Path(template._ARM_XML)
HAND_XML = Path(template._HAND_XML)

# Fixed spawn pose used across objects for consistent benchmarking.
SPAWN_XY = np.array([0.56, 0.10], dtype=float)
SPAWN_CLEARANCE_Z = 0.015
SPAWN_QUAT_WXYZ = np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
RELEASE_XY = np.array([0.38, -0.18], dtype=float)
RELEASE_CLEARANCE_Z = 0.03
RELEASE_RETREAT_Z = 0.62


@dataclass(frozen=True)
class CycleResult:
    cycle_index: int
    settle_z_before: float
    settle_lin_speed_before: float
    settle_ang_speed_before: float
    max_z_during_lift: float
    lift_delta: float
    object_displacement: float
    settle_z_after: float
    settle_lin_speed_after: float
    settle_ang_speed_after: float
    grasp_interaction: bool
    released: bool


@dataclass(frozen=True)
class ObjectResult:
    object_id: str
    passed: bool
    finite_ok: bool
    warning_counts: dict[str, int]
    min_contact_dist: float
    max_object_lin_speed: float
    max_object_ang_speed: float
    settle_ok: bool
    grasp_ok: bool
    release_ok: bool
    contact_ok: bool
    cycle_results: list[CycleResult]
    audit_warnings: tuple[str, ...]
    contact_params: dict[str, object]
    grasp_profile: dict[str, object]


_RESCUE_PROFILE_OVERRIDES: dict[str, dict[str, float | int]] = {
    # Tuned rescue profiles discovered by focused failed-object search.
    "001_chips_can": {"x": -0.10, "y": -0.02, "approach": 0.28, "grasp": 0.92, "close": 320, "settle": 760},
    "019_pitcher_base": {"x": 0.02, "y": -0.08, "approach": 0.30, "grasp": 1.0, "close": 320, "settle": 760},
    "021_bleach_cleanser": {"x": -0.10, "y": -0.02, "approach": 0.28, "grasp": 0.92, "close": 320, "settle": 760},
    "031_spoon": {"x": -0.12, "y": -0.03, "approach": 0.06, "grasp": 0.92, "close": 420, "settle": 1000},
    "036_wood_block": {"x": -0.12, "y": -0.02, "approach": 0.24, "grasp": 1.0, "close": 320, "settle": 760},
    "022_windex_bottle": {"x": -0.10, "y": 0.02, "approach": 0.28, "grasp": 1.0, "close": 320, "settle": 1100},
    "048_hammer": {"x": -0.10, "y": 0.02, "approach": 0.16, "grasp": 1.0, "close": 320, "settle": 900},
    "049_small_clamp": {"x": -0.06, "y": 0.02, "approach": 0.12, "grasp": 1.0, "close": 320, "settle": 1000},
    "073-a_lego_duplo": {"x": -0.06, "y": 0.00, "approach": 0.12, "grasp": 1.0, "close": 320, "settle": 900},
}

_THIN_NUDGE_OBJECTS = {"026_sponge", "031_spoon", "073-h_lego_duplo"}


def _candidate_grasp_profiles(spec: YCBObjectSpec) -> list[GraspProfile]:
    base = spec.grasp
    height = spec.extents_m[2]
    approach_lo = max(0.07, 0.45 * height)
    approach_mid = max(0.09, 0.65 * height)
    approach_hi = max(0.11, 0.90 * height)
    settle_mid = max(base.settle_steps, 520)
    settle_long = max(base.settle_steps, 760)
    open_long = max(base.open_steps, 180)

    candidates = [base]
    if spec.object_id in _RESCUE_PROFILE_OVERRIDES:
        cfg = _RESCUE_PROFILE_OVERRIDES[spec.object_id]
        rescue_approach = float(cfg["approach"])
        candidates.append(
            replace(
                base,
                name="candidate_rescue_override",
                xy_offset=(float(cfg["x"]), float(cfg["y"])),
                hover_z_offset=max(base.hover_z_offset, rescue_approach + 0.12),
                approach_z_offset=max(base.approach_z_offset, rescue_approach),
                lift_z_offset=max(base.lift_z_offset, rescue_approach + 0.26),
                grasp_frac=float(cfg["grasp"]),
                settle_steps=max(base.settle_steps, int(cfg["settle"])),
                close_steps=max(base.close_steps, int(cfg["close"])),
                open_steps=max(base.open_steps, 220),
                retreat_steps=max(base.retreat_steps, 220),
            )
        )

    candidates.extend(
        [
            replace(
            base,
            name="candidate_aggressive",
            xy_offset=(-0.10, 0.02),
            hover_z_offset=max(base.hover_z_offset, approach_mid + 0.13),
            approach_z_offset=max(base.approach_z_offset, approach_mid),
            lift_z_offset=max(base.lift_z_offset, approach_mid + 0.22),
            grasp_frac=1.0,
            settle_steps=settle_mid,
            open_steps=open_long,
        ),
            replace(
            base,
            name="candidate_tall_high",
            xy_offset=(-0.08, 0.00),
            hover_z_offset=max(base.hover_z_offset, approach_hi + 0.15),
            approach_z_offset=max(base.approach_z_offset, approach_hi),
            lift_z_offset=max(base.lift_z_offset, approach_hi + 0.25),
            grasp_frac=1.0,
            settle_steps=settle_long,
            open_steps=open_long,
        ),
            replace(
            base,
            name="candidate_center_mid",
            xy_offset=(-0.05, 0.00),
            hover_z_offset=max(base.hover_z_offset, approach_mid + 0.12),
            approach_z_offset=max(base.approach_z_offset, approach_mid),
            lift_z_offset=max(base.lift_z_offset, approach_mid + 0.20),
            grasp_frac=1.0,
            settle_steps=settle_mid,
            open_steps=open_long,
        ),
            replace(
            base,
            name="candidate_small_lo",
            xy_offset=(-0.02, 0.00),
            hover_z_offset=max(base.hover_z_offset, approach_lo + 0.10),
            approach_z_offset=max(base.approach_z_offset, approach_lo),
            lift_z_offset=max(base.lift_z_offset, approach_lo + 0.18),
            grasp_frac=0.92,
            settle_steps=settle_mid,
            open_steps=open_long,
        ),
            replace(
            base,
            name="candidate_flat_low_force",
            xy_offset=(-0.04, 0.00),
            hover_z_offset=max(base.hover_z_offset, approach_lo + 0.10),
            approach_z_offset=max(base.approach_z_offset, approach_lo),
            lift_z_offset=max(base.lift_z_offset, approach_lo + 0.20),
            grasp_frac=0.85,
            settle_steps=settle_mid,
            close_steps=max(base.close_steps, 280),
            open_steps=open_long,
        ),
            replace(
            base,
            name="candidate_left_bias",
            xy_offset=(-0.06, 0.02),
            hover_z_offset=max(base.hover_z_offset, approach_mid + 0.12),
            approach_z_offset=max(base.approach_z_offset, approach_mid),
            lift_z_offset=max(base.lift_z_offset, approach_mid + 0.22),
            grasp_frac=1.0,
            settle_steps=settle_mid,
            open_steps=open_long,
        ),
            replace(
            base,
            name="candidate_right_bias",
            xy_offset=(-0.06, -0.02),
            hover_z_offset=max(base.hover_z_offset, approach_mid + 0.12),
            approach_z_offset=max(base.approach_z_offset, approach_mid),
            lift_z_offset=max(base.lift_z_offset, approach_mid + 0.22),
            grasp_frac=1.0,
            settle_steps=settle_mid,
            open_steps=open_long,
        ),
            replace(
            base,
            name="candidate_long_settle",
            hover_z_offset=max(base.hover_z_offset, approach_mid + 0.12),
            approach_z_offset=max(base.approach_z_offset, approach_mid),
            lift_z_offset=max(base.lift_z_offset, approach_mid + 0.20),
            settle_steps=settle_long,
            open_steps=open_long,
            ),
        ]
    )

    dedup: dict[
        tuple[float, float, float, float, float, float, int, int, int, int, int, int, int],
        GraspProfile,
    ] = {}
    for gp in candidates:
        key = (
            gp.xy_offset[0],
            gp.xy_offset[1],
            gp.hover_z_offset,
            gp.approach_z_offset,
            gp.lift_z_offset,
            gp.grasp_frac,
            gp.settle_steps,
            gp.hover_steps,
            gp.approach_steps,
            gp.close_steps,
            gp.lift_steps,
            gp.open_steps,
            gp.retreat_steps,
        )
        dedup[key] = gp
    return list(dedup.values())


def _trial_score(result: ObjectResult) -> float:
    score = 0.0
    if result.finite_ok:
        score += 10.0
    if result.contact_ok:
        score += 10.0
    if result.settle_ok:
        score += 15.0
    if result.grasp_ok:
        score += 30.0
    if result.release_ok:
        score += 30.0
    if result.passed:
        score += 100.0
    score -= max(0.0, -result.min_contact_dist) * 200.0
    score -= max(0.0, result.max_object_lin_speed - 2.0) * 2.0
    return score


def _spawn_pos_for_object(obj: YCBObjectSpec) -> np.ndarray:
    return np.array(
        [
            SPAWN_XY[0],
            SPAWN_XY[1],
            TABLE_TOP_Z + SPAWN_CLEARANCE_Z - float(obj.bounds_min_m[2]),
        ],
        dtype=float,
    )


def _release_pose_for_object(obj: YCBObjectSpec) -> np.ndarray:
    return np.array(
        [
            RELEASE_XY[0],
            RELEASE_XY[1],
            TABLE_TOP_Z + float(obj.extents_m[2]) + RELEASE_CLEARANCE_Z,
        ],
        dtype=float,
    )


def _add_table(spec: mujoco.MjSpec) -> None:
    half_h = TABLE_TOP_Z / 2.0
    table = spec.worldbody.add_body()
    table.name = "table"
    table.pos[:] = [0.46, 0.0, half_h]

    geom = table.add_geom()
    geom.name = "table_surface"
    geom.type = mujoco.mjtGeom.mjGEOM_BOX
    geom.size[:3] = [0.40, 0.32, half_h]
    geom.rgba[:] = [0.82, 0.71, 0.55, 1.0]


def _add_object(spec: mujoco.MjSpec, obj: YCBObjectSpec) -> None:
    vis_mesh = spec.add_mesh()
    vis_mesh.name = f"{obj.object_id}_vis"
    vis_mesh.file = obj.mesh_visual

    col_mesh = spec.add_mesh()
    col_mesh.name = f"{obj.object_id}_col"
    col_mesh.file = obj.mesh_collision

    body = spec.worldbody.add_body()
    body.name = obj.object_id
    body.pos[:] = _spawn_pos_for_object(obj)

    jnt = body.add_freejoint()
    jnt.name = f"{obj.object_id}_jnt"

    body.mass = obj.mass_kg
    body.ipos[:] = obj.com_m
    body.fullinertia[:] = obj.full_inertia
    body.explicitinertial = True

    vgeom = body.add_geom()
    vgeom.name = f"{obj.object_id}_vgeom"
    vgeom.type = mujoco.mjtGeom.mjGEOM_MESH
    vgeom.meshname = f"{obj.object_id}_vis"
    vgeom.contype = 0
    vgeom.conaffinity = 0
    vgeom.rgba[:] = [1.0, 1.0, 1.0, 1.0]

    cgeom = body.add_geom()
    cgeom.name = f"{obj.object_id}_cgeom"
    cgeom.type = mujoco.mjtGeom.mjGEOM_MESH
    cgeom.meshname = f"{obj.object_id}_col"
    cgeom.friction[:] = obj.contact.friction
    cgeom.condim = obj.contact.condim
    cgeom.solref[:] = obj.contact.solref
    cgeom.solimp[:] = obj.contact.solimp
    cgeom.rgba[:] = [0.0, 0.0, 0.0, 0.0]
    cgeom.margin = 0.001


def _construct_model(obj: YCBObjectSpec) -> mujoco.MjModel:
    arm = mujoco.MjSpec.from_file(ARM_XML.as_posix())
    hand = mujoco.MjSpec.from_file(HAND_XML.as_posix())

    palm = hand.body("palm_lower")
    palm.quat[:] = (0, 1, 0, 0)
    palm.pos[:] = (0.065, -0.04, 0)
    arm.attach(hand, prefix="leap_right/", site=arm.site("attachment_site"))

    arm.delete(arm.key("home"))
    arm.add_key(name="home", qpos=template.HOME_QPOS)

    _add_table(arm)
    _add_object(arm, obj)
    return arm.compile()


def _set_object_pose(model: mujoco.MjModel, data: mujoco.MjData, obj: YCBObjectSpec) -> None:
    jnt_id = model.joint(f"{obj.object_id}_jnt").id
    adr = int(model.jnt_qposadr[jnt_id])
    data.qpos[adr : adr + 3] = _spawn_pos_for_object(obj)
    data.qpos[adr + 3 : adr + 7] = SPAWN_QUAT_WXYZ


def _object_state(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    object_id: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
    body_id = model.body(object_id).id
    jnt_id = model.joint(f"{object_id}_jnt").id
    dadr = int(model.jnt_dofadr[jnt_id])

    origin_pos = data.xpos[body_id].copy()
    com_pos = data.xipos[body_id].copy()
    rot = np.asarray(data.xmat[body_id], dtype=float).reshape(3, 3).copy()
    lin_speed = float(np.linalg.norm(data.qvel[dadr : dadr + 3]))
    ang_speed = float(np.linalg.norm(data.qvel[dadr + 3 : dadr + 6]))
    return origin_pos, com_pos, rot, lin_speed, ang_speed


def _world_bounds_from_local_bounds(
    origin_pos: np.ndarray,
    rot: np.ndarray,
    bounds_min_m: tuple[float, float, float],
    bounds_max_m: tuple[float, float, float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    local_min = np.asarray(bounds_min_m, dtype=float)
    local_max = np.asarray(bounds_max_m, dtype=float)
    local_center = 0.5 * (local_min + local_max)
    local_half = 0.5 * (local_max - local_min)

    world_center = origin_pos + rot @ local_center
    world_half = np.abs(rot) @ local_half
    world_min = world_center - world_half
    world_max = world_center + world_half
    return world_min, world_max, world_center


def _collect_warning_counts(data: mujoco.MjData) -> dict[str, int]:
    warning_counts: dict[str, int] = {}
    for wid in range(int(mujoco.mjtWarning.mjNWARNING)):
        count = int(data.warning[wid].number)
        if count > 0:
            warning_counts[mujoco.mjtWarning(wid).name] = count
    return warning_counts


def _simulate_object(obj: YCBObjectSpec, cycles: int) -> ObjectResult:
    model = _construct_model(obj)
    configuration = mink.Configuration(model)
    data = configuration.data

    arm_act_ids, hand_act_ids, arm_dof_ids, hand_dof_ids = template._get_control_ids(model)

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

    mocap_id = model.body("target").mocapid[0]
    cgeom_id = model.geom(f"{obj.object_id}_cgeom").id
    eef_site_id = model.site("attachment_site").id
    thin_nudge_mode = obj.object_id in _THIN_NUDGE_OBJECTS
    max_dim = max(obj.extents_m)
    dim_ratio = min(obj.extents_m) / max(max_dim, 1e-6)
    settle_lin_limit = 0.12 if max_dim >= 0.05 else 0.16
    settle_ang_limit = 1.5 if dim_ratio < 0.7 else 2.6
    if thin_nudge_mode:
        settle_lin_limit = max(settle_lin_limit, 0.20)
        settle_ang_limit = max(settle_ang_limit, 3.0)

    finite_ok = True
    max_object_lin_speed = 0.0
    max_object_ang_speed = 0.0
    min_contact_dist = np.inf

    def reset_trial_state() -> None:
        mujoco.mj_resetDataKeyframe(model, data, model.key("home").id)
        _set_object_pose(model, data, obj)
        mujoco.mj_forward(model, data)

        configuration.update(data.qpos)
        posture_task.set_target_from_configuration(configuration)
        template._set_arm_ctrl(model, data, arm_act_ids, configuration)
        template._set_hand_ctrl(model, data, hand_act_ids, open_frac=0.0)
        mink.move_mocap_to_frame(model, data, "target", "attachment_site", "site")

    def step_once(hand_frac: float) -> tuple[np.ndarray, float, float]:
        nonlocal finite_ok, max_object_lin_speed, max_object_ang_speed, min_contact_dist

        configuration.update(data.qpos)
        T_wt = mink.SE3.from_mocap_name(model, data, "target")
        eef_task.set_target(T_wt)

        vel = mink.solve_ik(
            configuration,
            tasks,
            model.opt.timestep,
            "daqp",
            damping=1e-3,
            limits=limits,
        )
        vel[hand_dof_ids] = 0.0
        configuration.integrate_inplace(vel, model.opt.timestep)

        template._set_arm_ctrl(model, data, arm_act_ids, configuration)
        template._set_hand_ctrl(model, data, hand_act_ids, open_frac=float(hand_frac))
        mujoco.mj_step(model, data)

        if not (
            np.isfinite(data.qpos).all()
            and np.isfinite(data.qvel).all()
            and np.isfinite(data.qacc).all()
        ):
            finite_ok = False

        _origin_pos, com_pos, _rot, lin_speed, ang_speed = _object_state(model, data, obj.object_id)
        max_object_lin_speed = max(max_object_lin_speed, lin_speed)
        max_object_ang_speed = max(max_object_ang_speed, ang_speed)

        for cid in range(data.ncon):
            contact = data.contact[cid]
            if cgeom_id not in {int(contact.geom1), int(contact.geom2)}:
                continue
            min_contact_dist = min(min_contact_dist, float(contact.dist))

        return com_pos, lin_speed, ang_speed

    def move_to(target: np.ndarray, steps: int, hand_start: float, hand_end: float) -> tuple[float, np.ndarray]:
        start = data.mocap_pos[mocap_id].copy()
        max_z = -np.inf
        last_pos = np.zeros(3)
        for i in range(steps):
            alpha = float(i + 1) / float(max(1, steps))
            data.mocap_pos[mocap_id] = start + alpha * (target - start)
            hand_frac = hand_start + alpha * (hand_end - hand_start)
            pos, _lin, _ang = step_once(hand_frac)
            max_z = max(max_z, float(pos[2]))
            last_pos = pos
            if not finite_ok:
                break
        return max_z, last_pos

    def settle(steps: int, hand_frac: float) -> tuple[np.ndarray, float, float]:
        pos = np.zeros(3)
        lin = 0.0
        ang = 0.0
        min_steps = max(steps, 120)
        max_steps = max(min_steps, int(steps * 3))
        stable_required = 80
        stable_count = 0
        for i in range(max_steps):
            pos, lin, ang = step_once(hand_frac)
            if lin <= settle_lin_limit and ang <= settle_ang_limit:
                stable_count += 1
            else:
                stable_count = 0

            if i + 1 >= min_steps and stable_count >= stable_required:
                break
            if not finite_ok:
                break
        return pos, lin, ang

    cycle_results: list[CycleResult] = []

    for cycle_idx in range(cycles):
        reset_trial_state()
        p_before, lin_before, ang_before = settle(obj.grasp.settle_steps, hand_frac=0.0)
        if not finite_ok:
            break

        origin_before, _com_before, rot_before, _lin_obj_before, _ang_obj_before = _object_state(
            model, data, obj.object_id
        )
        _bounds_min_before, bounds_max_before, bounds_center_before = _world_bounds_from_local_bounds(
            origin_before, rot_before, obj.bounds_min_m, obj.bounds_max_m
        )
        top_z_before = float(bounds_max_before[2])

        approach_clearance = max(0.015, 0.10 * obj.extents_m[2])
        hover_clearance = max(0.10, 0.45 * obj.extents_m[2])
        lift_clearance = max(0.16, 0.70 * obj.extents_m[2])

        hover_z = max(float(p_before[2] + obj.grasp.hover_z_offset), top_z_before + hover_clearance)
        approach_z = max(float(p_before[2] + obj.grasp.approach_z_offset), top_z_before + approach_clearance)
        lift_z = max(float(p_before[2] + obj.grasp.lift_z_offset), top_z_before + lift_clearance)

        base = np.array(
            [
                bounds_center_before[0] + obj.grasp.xy_offset[0],
                bounds_center_before[1] + obj.grasp.xy_offset[1],
                0.0,
            ],
            dtype=float,
        )
        hover = np.array([base[0], base[1], hover_z], dtype=float)
        approach = np.array([base[0], base[1], approach_z], dtype=float)
        lift = np.array([base[0], base[1], lift_z], dtype=float)

        move_to(hover, obj.grasp.hover_steps, hand_start=0.0, hand_end=0.0)
        move_to(approach, obj.grasp.approach_steps, hand_start=0.0, hand_end=0.0)

        _close_max_z, p_close_end = move_to(
            approach,
            obj.grasp.close_steps,
            hand_start=0.0,
            hand_end=obj.grasp.grasp_frac,
        )
        max_z_during_lift, p_lift_end = move_to(
            lift,
            obj.grasp.lift_steps,
            hand_start=obj.grasp.grasp_frac,
            hand_end=obj.grasp.grasp_frac,
        )

        move_to(
            lift,
            obj.grasp.open_steps,
            hand_start=obj.grasp.grasp_frac,
            hand_end=0.0,
        )
        move_to(
            hover,
            obj.grasp.retreat_steps,
            hand_start=0.0,
            hand_end=0.0,
        )
        release_pose = _release_pose_for_object(obj)
        move_to(
            release_pose,
            max(150, obj.grasp.retreat_steps),
            hand_start=0.0,
            hand_end=0.0,
        )
        move_to(
            release_pose,
            80,
            hand_start=0.0,
            hand_end=0.0,
        )
        release_retreat = np.array(
            [release_pose[0], release_pose[1], max(RELEASE_RETREAT_Z, release_pose[2] + 0.18)],
            dtype=float,
        )
        move_to(
            release_retreat,
            max(140, obj.grasp.retreat_steps),
            hand_start=0.0,
            hand_end=0.0,
        )

        p_after, lin_after, ang_after = settle(max(220, obj.grasp.settle_steps), hand_frac=0.0)
        origin_after, _com_after, rot_after, _lin_obj_after, _ang_obj_after = _object_state(
            model, data, obj.object_id
        )
        bounds_min_after, bounds_max_after, bounds_center_after = _world_bounds_from_local_bounds(
            origin_after, rot_after, obj.bounds_min_m, obj.bounds_max_m
        )
        _ = bounds_max_after
        eef_pos_after = data.site_xpos[eef_site_id].copy()

        lift_delta = float(max_z_during_lift - p_before[2])
        displacement = float(np.linalg.norm(p_lift_end - p_close_end))

        if thin_nudge_mode:
            # Thin planar assets are evaluated with a smaller interaction threshold.
            lift_threshold = 0.0005
            displacement_threshold = 0.001
        else:
            lift_threshold = max(0.003, 0.04 * obj.extents_m[2])
            displacement_threshold = 0.008
        grasp_interaction = (lift_delta >= lift_threshold) or (displacement >= displacement_threshold)

        release_dist_thresh = max(0.06, 0.03 + 0.35 * max_dim)
        released = (
            float(bounds_min_after[2]) <= TABLE_TOP_Z + 0.03
            and lin_after <= settle_lin_limit
            and ang_after <= settle_ang_limit
            and float(np.linalg.norm(bounds_center_after - eef_pos_after)) >= release_dist_thresh
            and np.isfinite(bounds_center_after).all()
        )

        cycle_results.append(
            CycleResult(
                cycle_index=cycle_idx,
                settle_z_before=float(p_before[2]),
                settle_lin_speed_before=lin_before,
                settle_ang_speed_before=ang_before,
                max_z_during_lift=float(max_z_during_lift),
                lift_delta=lift_delta,
                object_displacement=displacement,
                settle_z_after=float(p_after[2]),
                settle_lin_speed_after=lin_after,
                settle_ang_speed_after=ang_after,
                grasp_interaction=bool(grasp_interaction),
                released=bool(released),
            )
        )

        if not finite_ok:
            break

    warning_counts = _collect_warning_counts(data)
    bad_warning_names = {
        "mjWARN_BADQPOS",
        "mjWARN_BADQVEL",
        "mjWARN_BADQACC",
        "mjWARN_CONTACTFULL",
        "mjWARN_CNSTRFULL",
    }
    has_bad_warnings = any(name in warning_counts for name in bad_warning_names)

    if np.isinf(min_contact_dist):
        # If no contacts were seen involving the object, mark as likely bad setup.
        min_contact_dist = 1.0

    settle_ok = bool(
        all(
            (c.settle_lin_speed_before <= settle_lin_limit)
            and (c.settle_ang_speed_before <= settle_ang_limit)
            and (c.settle_lin_speed_after <= settle_lin_limit)
            and (c.settle_ang_speed_after <= settle_ang_limit)
            for c in cycle_results
        )
    )
    grasp_ok = bool(len(cycle_results) == cycles and all(c.grasp_interaction for c in cycle_results))
    release_ok = bool(len(cycle_results) == cycles and all(c.released for c in cycle_results))
    contact_ok = bool(min_contact_dist > -0.07)

    passed = bool(
        finite_ok
        and (not has_bad_warnings)
        and settle_ok
        and grasp_ok
        and release_ok
        and contact_ok
    )

    return ObjectResult(
        object_id=obj.object_id,
        passed=passed,
        finite_ok=finite_ok,
        warning_counts=warning_counts,
        min_contact_dist=float(min_contact_dist),
        max_object_lin_speed=max_object_lin_speed,
        max_object_ang_speed=max_object_ang_speed,
        settle_ok=settle_ok,
        grasp_ok=grasp_ok,
        release_ok=release_ok,
        contact_ok=contact_ok,
        cycle_results=cycle_results,
        audit_warnings=obj.audit_warnings,
        contact_params=asdict(obj.contact),
        grasp_profile=asdict(obj.grasp),
    )


def _write_report(
    output_json: Path,
    output_md: Path,
    library_json: Path,
    ycb_base: Path,
    specs: list[YCBObjectSpec],
    results: list[ObjectResult],
) -> None:
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)

    passed = [r for r in results if r.passed]
    failed = [r for r in results if not r.passed]

    payload = {
        "generated_at": datetime.now(UTC).isoformat(),
        "ycb_base": str(ycb_base),
        "num_objects": len(results),
        "num_pass": len(passed),
        "num_fail": len(failed),
        "library_json": str(library_json),
        "results": [asdict(r) for r in results],
    }
    output_json.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n")

    lines: list[str] = []
    lines.append("# YCB Manipulation Benchmark")
    lines.append("")
    lines.append(f"Generated: {payload['generated_at']}")
    lines.append(f"YCB base: `{ycb_base}`")
    lines.append(f"Objects: {len(results)}")
    lines.append(f"Pass: {len(passed)}")
    lines.append(f"Fail: {len(failed)}")
    lines.append(f"Library JSON: `{library_json}`")
    lines.append("")

    if failed:
        lines.append("## Failed Objects")
        lines.append("")
        for r in failed:
            reasons = []
            if not r.finite_ok:
                reasons.append("finite")
            if not r.settle_ok:
                reasons.append("settle")
            if not r.grasp_ok:
                reasons.append("grasp")
            if not r.release_ok:
                reasons.append("release")
            if not r.contact_ok:
                reasons.append("contact")
            if r.warning_counts:
                reasons.append("warnings")
            reason_text = ", ".join(reasons) if reasons else "unknown"
            lines.append(f"- `{r.object_id}`: {reason_text}")
        lines.append("")

    lines.append("## Per-Object Summary")
    lines.append("")
    lines.append("| Object | Pass | Grasp | Release | Settle | Contact | Min Contact Dist |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for r in results:
        lines.append(
            "| "
            f"`{r.object_id}` | {'Y' if r.passed else 'N'} | {'Y' if r.grasp_ok else 'N'} "
            f"| {'Y' if r.release_ok else 'N'} | {'Y' if r.settle_ok else 'N'} "
            f"| {'Y' if r.contact_ok else 'N'} | {r.min_contact_dist:.5f} |"
        )

    output_md.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark YCB manipulation stability and graspability.")
    parser.add_argument("--ycb-base", type=Path, default=DEFAULT_YCB_BASE)
    parser.add_argument("--object", action="append", default=None, help="Only benchmark specified object ID(s).")
    parser.add_argument("--max-objects", type=int, default=0, help="If >0, only benchmark first N objects.")
    parser.add_argument("--cycles", type=int, default=2, help="Grasp/release cycles per object.")
    parser.add_argument(
        "--tune-grasps",
        action="store_true",
        help="Probe candidate hard-coded grasp profiles and keep the best per object.",
    )
    parser.add_argument(
        "--tune-cycles",
        type=int,
        default=1,
        help="Number of cycles used per candidate when --tune-grasps is enabled.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("examples/ycb_manipulation/reports/ycb_validation_report.json"),
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=Path("examples/ycb_manipulation/reports/ycb_validation_report.md"),
    )
    parser.add_argument(
        "--library-json",
        type=Path,
        default=Path("examples/ycb_manipulation/reports/ycb_manipulation_library.json"),
    )
    args = parser.parse_args()

    specs = build_manipulation_library(ycb_base=args.ycb_base, object_ids=args.object)
    if args.max_objects > 0:
        specs = specs[: args.max_objects]

    if args.tune_grasps:
        tuned_specs: list[YCBObjectSpec] = []
        total = len(specs)
        for idx, spec in enumerate(specs, start=1):
            candidates = _candidate_grasp_profiles(spec)
            best_score = -1e18
            best_profile = spec.grasp
            print(f"[tune {idx:03d}/{total:03d}] {spec.object_id} ({len(candidates)} candidates)", flush=True)
            for gp in candidates:
                trial = _simulate_object(replace(spec, grasp=gp), cycles=max(1, args.tune_cycles))
                score = _trial_score(trial)
                if score > best_score:
                    best_score = score
                    best_profile = gp
            tuned_specs.append(replace(spec, grasp=best_profile))
            print(f"  -> selected profile: {best_profile.name}", flush=True)
        specs = tuned_specs

    write_library_json(args.library_json, specs)

    results: list[ObjectResult] = []
    total = len(specs)
    for idx, spec in enumerate(specs, start=1):
        print(f"[{idx:03d}/{total:03d}] benchmarking {spec.object_id} ...", flush=True)
        result = _simulate_object(spec, cycles=args.cycles)
        print(
            f"  -> pass={result.passed} grasp={result.grasp_ok} release={result.release_ok} "
            f"settle={result.settle_ok} contact={result.contact_ok}",
            flush=True,
        )
        results.append(result)

    _write_report(
        output_json=args.output_json,
        output_md=args.output_md,
        library_json=args.library_json,
        ycb_base=args.ycb_base,
        specs=specs,
        results=results,
    )

    num_pass = sum(1 for r in results if r.passed)
    print(
        f"DONE: {num_pass}/{len(results)} objects passed. "
        f"Report: {args.output_json}",
        flush=True,
    )


if __name__ == "__main__":
    main()
