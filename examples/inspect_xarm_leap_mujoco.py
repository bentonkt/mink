from __future__ import annotations

import math
from pathlib import Path

import mujoco
import numpy as np

HERE = Path(__file__).parent
ARM_XML = HERE / "ufactory_xarm7" / "scene.xml"
HAND_XML = HERE / "leap_hand" / "right_hand.xml"

# fmt: off
HOME_QPOS = np.array([
    # xArm.
    0.0, -0.247, 0.0, 0.909, 0.0, 1.15644, 0.0,
    # LEAP hand.
    0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0,
], dtype=float)
# fmt: on


def obj_name(model: mujoco.MjModel, obj: mujoco.mjtObj, obj_id: int) -> str:
    name = mujoco.mj_id2name(model, obj, obj_id)
    if name is not None:
        return name
    return f"{obj.name.lower()}[{obj_id}]"


def build_combined_model(*, add_probe_object: bool) -> mujoco.MjModel:
    arm = mujoco.MjSpec.from_file(ARM_XML.as_posix())
    hand = mujoco.MjSpec.from_file(HAND_XML.as_posix())

    # Align and attach the hand to the xArm attachment site.
    palm = hand.body("palm_lower")
    palm.quat[:] = (0, 1, 0, 0)
    palm.pos[:] = (0.065, -0.04, 0)
    arm.attach(hand, prefix="leap_right/", site=arm.site("attachment_site"))

    # Replace the home key with a combined home posture.
    home_key = arm.key("home")
    arm.delete(home_key)
    arm.add_key(name="home", qpos=HOME_QPOS.tolist())

    if add_probe_object:
        probe_body = arm.worldbody.add_body(name="probe_object", pos=(0.0, 0.0, 0.5))
        probe_body.add_geom(
            name="probe_object_geom",
            type=mujoco.mjtGeom.mjGEOM_SPHERE,
            size=(0.016, 0.016, 0.016),
            rgba=(0.2, 0.7, 0.9, 1.0),
        )

    return arm.compile()


def chain_from_body(model: mujoco.MjModel, body_name: str) -> list[str]:
    body_id = model.body(body_name).id
    chain: list[str] = []

    while True:
        name = obj_name(model, mujoco.mjtObj.mjOBJ_BODY, body_id)
        jnt_adr = model.body_jntadr[body_id]
        jnt_num = model.body_jntnum[body_id]
        if jnt_num > 0:
            jnames = [
                obj_name(model, mujoco.mjtObj.mjOBJ_JOINT, jid)
                for jid in range(jnt_adr, jnt_adr + jnt_num)
            ]
            chain.append(f"{name} [{', '.join(jnames)}]")
        else:
            chain.append(name)

        if body_id == 0:
            break
        body_id = model.body_parentid[body_id]

    chain.reverse()
    return chain


def print_chains(model: mujoco.MjModel) -> None:
    print("\n== Kinematic Chains ==")
    xarm_chain = chain_from_body(model, "link7")
    print("xArm chain to tool:")
    print("  " + " -> ".join(xarm_chain))

    leap_tip_bodies = [
        "leap_right/fingertip",
        "leap_right/fingertip_2",
        "leap_right/fingertip_3",
        "leap_right/thumb_fingertip",
    ]
    for body in leap_tip_bodies:
        print(f"{body} chain:")
        print("  " + " -> ".join(chain_from_body(model, body)))


def actuator_target(model: mujoco.MjModel, actuator_id: int) -> str:
    trn_type = int(model.actuator_trntype[actuator_id])
    trn_id = int(model.actuator_trnid[actuator_id, 0])

    if trn_type == int(mujoco.mjtTrn.mjTRN_JOINT):
        return f"joint:{obj_name(model, mujoco.mjtObj.mjOBJ_JOINT, trn_id)}"
    if trn_type == int(mujoco.mjtTrn.mjTRN_TENDON):
        return f"tendon:{obj_name(model, mujoco.mjtObj.mjOBJ_TENDON, trn_id)}"
    return f"trn_type={trn_type}, id={trn_id}"


def print_actuators(model: mujoco.MjModel) -> None:
    print("\n== Actuators ==")
    for aid in range(model.nu):
        name = obj_name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, aid)
        ctrl_limited = bool(model.actuator_ctrllimited[aid])
        force_limited = bool(model.actuator_forcelimited[aid])
        ctrl_range = tuple(float(x) for x in model.actuator_ctrlrange[aid])
        force_range = tuple(float(x) for x in model.actuator_forcerange[aid])
        ctrl_text = f"{ctrl_range}" if ctrl_limited else "unbounded"
        force_text = f"{force_range}" if force_limited else "unbounded"
        print(
            f"- {name}: {actuator_target(model, aid)}, "
            f"ctrl={ctrl_text}, force={force_text}"
        )


def print_joint_limits(model: mujoco.MjModel) -> None:
    print("\n== Joint Limits ==")
    for jid in range(model.njnt):
        if not model.jnt_limited[jid]:
            continue
        name = obj_name(model, mujoco.mjtObj.mjOBJ_JOINT, jid)
        low, high = model.jnt_range[jid]
        print(f"- {name}: [{low:.4f}, {high:.4f}]")


def print_contact_setup(model: mujoco.MjModel) -> None:
    print("\n== Contact Geometries ==")
    collidable_by_body: dict[str, int] = {}
    for gid in range(model.ngeom):
        if int(model.geom_contype[gid]) == 0 or int(model.geom_conaffinity[gid]) == 0:
            continue
        body_id = int(model.geom_bodyid[gid])
        body_name = obj_name(model, mujoco.mjtObj.mjOBJ_BODY, body_id)
        collidable_by_body[body_name] = collidable_by_body.get(body_name, 0) + 1

    for body_name in sorted(collidable_by_body):
        print(f"- {body_name}: {collidable_by_body[body_name]} collidable geom(s)")

    print("\nContact excludes:")
    for signature in model.exclude_signature:
        sig = int(signature)
        body1_id = sig & 0xFFFF
        body2_id = sig >> 16
        body1 = obj_name(model, mujoco.mjtObj.mjOBJ_BODY, body1_id)
        body2 = obj_name(model, mujoco.mjtObj.mjOBJ_BODY, body2_id)
        print(f"- {body1} <-> {body2}")


def run_arm_stability_check(model: mujoco.MjModel) -> tuple[bool, dict[str, object]]:
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, model.key("home").id)

    arm_actuators = [
        aid
        for aid in range(model.nu)
        if not obj_name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, aid).startswith("leap_right/")
    ]
    if len(arm_actuators) < 7:
        raise RuntimeError("Expected at least 7 arm actuators.")

    arm_actuators = arm_actuators[:7]
    arm_home = np.array(HOME_QPOS[:7], dtype=float)
    amplitudes = np.array([0.18, 0.20, 0.18, 0.16, 0.16, 0.14, 0.14], dtype=float)
    phases = np.linspace(0.0, math.pi, num=7)

    duration = 6.0
    steps = int(duration / model.opt.timestep)
    finite_ok = True
    max_abs_qvel = 0.0
    max_abs_qacc = 0.0

    for step in range(steps):
        t = step * model.opt.timestep
        target = arm_home + amplitudes * np.sin(2.0 * math.pi * 0.25 * t + phases)

        for k, aid in enumerate(arm_actuators):
            if model.actuator_ctrllimited[aid]:
                low, high = model.actuator_ctrlrange[aid]
                data.ctrl[aid] = np.clip(target[k], low, high)
            else:
                data.ctrl[aid] = target[k]

        mujoco.mj_step(model, data)
        if not (
            np.isfinite(data.qpos).all()
            and np.isfinite(data.qvel).all()
            and np.isfinite(data.qacc).all()
        ):
            finite_ok = False
            break

        max_abs_qvel = max(max_abs_qvel, float(np.max(np.abs(data.qvel))))
        max_abs_qacc = max(max_abs_qacc, float(np.max(np.abs(data.qacc))))

    warning_counts: dict[str, int] = {}
    for wid in range(int(mujoco.mjtWarning.mjNWARNING)):
        count = int(data.warning[wid].number)
        if count > 0:
            warning_counts[mujoco.mjtWarning(wid).name] = count

    bad_warnings = any(
        name in warning_counts
        for name in ("mjWARN_BADQPOS", "mjWARN_BADQVEL", "mjWARN_BADQACC")
    )
    stable = finite_ok and not bad_warnings
    diagnostics: dict[str, object] = {
        "duration_sec": duration,
        "finite_ok": finite_ok,
        "max_abs_qvel": max_abs_qvel,
        "max_abs_qacc": max_abs_qacc,
        "warnings": warning_counts,
    }
    return stable, diagnostics


def leap_related_contacts(
    model: mujoco.MjModel, data: mujoco.MjData, object_geom_id: int
) -> tuple[int, set[str]]:
    count = 0
    bodies: set[str] = set()

    for cid in range(data.ncon):
        contact = data.contact[cid]
        if object_geom_id not in {contact.geom1, contact.geom2}:
            continue

        other_geom_id = contact.geom2 if contact.geom1 == object_geom_id else contact.geom1
        body_id = int(model.geom_bodyid[other_geom_id])
        body_name = obj_name(model, mujoco.mjtObj.mjOBJ_BODY, body_id)
        if body_name.startswith("leap_right/"):
            count += 1
            bodies.add(body_name)

    return count, bodies


def run_hand_contact_check(model: mujoco.MjModel) -> tuple[bool, dict[str, object]]:
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, model.key("home").id)
    mujoco.mj_forward(model, data)

    hand_actuators = [
        aid
        for aid in range(model.nu)
        if obj_name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, aid).startswith("leap_right/")
    ]
    if len(hand_actuators) != 16:
        raise RuntimeError(f"Expected 16 hand actuators, got {len(hand_actuators)}")

    hand_joint_ids = [
        jid
        for jid in range(model.njnt)
        if obj_name(model, mujoco.mjtObj.mjOBJ_JOINT, jid).startswith("leap_right/")
    ]
    arm_actuators = [
        aid
        for aid in range(model.nu)
        if not obj_name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, aid).startswith("leap_right/")
    ][:7]
    arm_home = np.array(HOME_QPOS[:7], dtype=float)

    probe_body_id = model.body("probe_object").id
    probe_geom_id = model.geom("probe_object_geom").id
    palm_body_id = model.body("leap_right/palm_lower").id
    tip_site_ids = [
        model.site("leap_right/tip_1").id,
        model.site("leap_right/tip_2").id,
        model.site("leap_right/tip_3").id,
        model.site("leap_right/th_tip").id,
    ]

    tip_centroid = np.mean([data.site_xpos[sid] for sid in tip_site_ids], axis=0)
    palm_pos = data.xpos[palm_body_id]
    model.body_pos[probe_body_id] = 0.7 * tip_centroid + 0.3 * palm_pos
    mujoco.mj_forward(model, data)

    for aid in hand_actuators:
        low, high = model.actuator_ctrlrange[aid]
        data.ctrl[aid] = np.clip(0.0, low, high)

    pre_contact = 0
    for _ in range(200):
        for k, aid in enumerate(arm_actuators):
            data.ctrl[aid] = arm_home[k]
        mujoco.mj_step(model, data)
        num, _ = leap_related_contacts(model, data, probe_geom_id)
        pre_contact = max(pre_contact, num)

    qpos_before = data.qpos.copy()

    max_contact = 0
    first_contact_step = -1
    touching_bodies: set[str] = set()
    for step in range(900):
        for k, aid in enumerate(arm_actuators):
            data.ctrl[aid] = arm_home[k]
        for aid in hand_actuators:
            low, high = model.actuator_ctrlrange[aid]
            data.ctrl[aid] = low + 0.9 * (high - low)
        mujoco.mj_step(model, data)

        num, bodies = leap_related_contacts(model, data, probe_geom_id)
        touching_bodies.update(bodies)
        max_contact = max(max_contact, num)
        if first_contact_step < 0 and num > 0:
            first_contact_step = step

    qpos_after = data.qpos.copy()
    max_joint_motion = 0.0
    for jid in hand_joint_ids:
        qadr = model.jnt_qposadr[jid]
        max_joint_motion = max(
            max_joint_motion, float(abs(qpos_after[qadr] - qpos_before[qadr]))
        )

    success = max_contact > 0 and max_joint_motion > 0.05
    diagnostics: dict[str, object] = {
        "max_contact_points": max_contact,
        "first_contact_step_after_closing": first_contact_step,
        "touching_bodies": sorted(touching_bodies),
        "max_hand_joint_motion_rad": max_joint_motion,
        "pre_contact_max_points": pre_contact,
    }
    return success, diagnostics


def main() -> None:
    model = build_combined_model(add_probe_object=False)

    print_chains(model)
    print_actuators(model)
    print_joint_limits(model)
    print_contact_setup(model)

    stable_ok, stable_diag = run_arm_stability_check(model)
    print("\n== Arm Stability Check ==")
    print(f"PASS={stable_ok}")
    for key, value in stable_diag.items():
        print(f"- {key}: {value}")

    contact_model = build_combined_model(add_probe_object=True)
    contact_ok, contact_diag = run_hand_contact_check(contact_model)
    print("\n== Hand Actuation + Contact Check ==")
    print(f"PASS={contact_ok}")
    for key, value in contact_diag.items():
        print(f"- {key}: {value}")

    if not (stable_ok and contact_ok):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
