from __future__ import annotations

import json
import xml.etree.ElementTree as et
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

DEFAULT_YCB_BASE = Path("/Users/bentontameling/Dev/ycb-tools/models/ycb")
TABLE_TOP_Z = 0.30


@dataclass(frozen=True)
class ContactParams:
    friction: tuple[float, float, float]
    condim: int = 6
    solref: tuple[float, float] = (0.02, 1.0)
    solimp: tuple[float, float, float, float, float] = (0.9, 0.95, 0.001, 0.5, 2.0)


@dataclass(frozen=True)
class GraspProfile:
    name: str
    xy_offset: tuple[float, float]
    hover_z_offset: float
    approach_z_offset: float
    lift_z_offset: float
    grasp_frac: float
    settle_steps: int
    hover_steps: int
    approach_steps: int
    close_steps: int
    lift_steps: int
    open_steps: int
    retreat_steps: int


@dataclass(frozen=True)
class YCBObjectSpec:
    object_id: str
    mass_kg: float
    com_m: tuple[float, float, float]
    inertia_diag: tuple[float, float, float]
    full_inertia: tuple[float, float, float, float, float, float]
    mesh_visual: str
    mesh_collision: str
    bounds_min_m: tuple[float, float, float]
    bounds_max_m: tuple[float, float, float]
    extents_m: tuple[float, float, float]
    contact: ContactParams
    grasp: GraspProfile
    audit_warnings: tuple[str, ...]


_GRASP_PROFILES: dict[str, GraspProfile] = {
    "tall": GraspProfile(
        name="tall",
        xy_offset=(-0.08, 0.00),
        hover_z_offset=0.25,
        approach_z_offset=0.12,
        lift_z_offset=0.36,
        grasp_frac=1.0,
        settle_steps=450,
        hover_steps=220,
        approach_steps=220,
        close_steps=260,
        lift_steps=260,
        open_steps=140,
        retreat_steps=180,
    ),
    "flat": GraspProfile(
        name="flat",
        xy_offset=(-0.04, 0.00),
        hover_z_offset=0.23,
        approach_z_offset=0.09,
        lift_z_offset=0.30,
        grasp_frac=0.85,
        settle_steps=420,
        hover_steps=220,
        approach_steps=200,
        close_steps=240,
        lift_steps=230,
        open_steps=120,
        retreat_steps=180,
    ),
    "small": GraspProfile(
        name="small",
        xy_offset=(-0.02, 0.00),
        hover_z_offset=0.22,
        approach_z_offset=0.10,
        lift_z_offset=0.30,
        grasp_frac=0.90,
        settle_steps=360,
        hover_steps=220,
        approach_steps=210,
        close_steps=260,
        lift_steps=230,
        open_steps=120,
        retreat_steps=180,
    ),
    "default": GraspProfile(
        name="default",
        xy_offset=(-0.04, 0.00),
        hover_z_offset=0.25,
        approach_z_offset=0.12,
        lift_z_offset=0.34,
        grasp_frac=1.0,
        settle_steps=420,
        hover_steps=220,
        approach_steps=220,
        close_steps=260,
        lift_steps=250,
        open_steps=140,
        retreat_steps=180,
    ),
}

# Object-specific hard-coded grasp overrides used ahead of geometry heuristics.
_GRASP_OVERRIDES: dict[str, str] = {
    "006_mustard_bottle": "tall",
    "025_mug": "default",
    "009_gelatin_box": "flat",
    "003_cracker_box": "tall",
    "004_sugar_box": "tall",
    "005_tomato_soup_can": "default",
    "001_chips_can": "tall",
}

# Thin/concave assets that benefit from softer contact and lower friction.
_CONTACT_OVERRIDES: dict[str, ContactParams] = {
    "031_spoon": ContactParams(
        friction=(0.7, 0.01, 0.001),
        condim=6,
        solref=(0.03, 1.0),
        solimp=(0.88, 0.95, 0.002, 0.5, 2.0),
    ),
    "049_small_clamp": ContactParams(
        friction=(0.8, 0.02, 0.001),
        condim=6,
        solref=(0.03, 1.0),
        solimp=(0.88, 0.95, 0.002, 0.5, 2.0),
    ),
    "072-h_toy_airplane": ContactParams(
        friction=(0.8, 0.02, 0.001),
        condim=6,
        solref=(0.03, 1.0),
        solimp=(0.88, 0.95, 0.002, 0.5, 2.0),
    ),
}


def discover_object_ids(ycb_base: Path = DEFAULT_YCB_BASE) -> list[str]:
    return sorted({p.parent.name for p in ycb_base.glob("*/*_physics.json")})


def _read_full_inertia(xml_path: Path) -> tuple[float, float, float, float, float, float]:
    root = et.parse(xml_path).getroot()
    inertial = root.find(".//inertial")
    if inertial is None:
        raise ValueError(f"No inertial element in {xml_path}")
    full = inertial.get("fullinertia")
    if not full:
        raise ValueError(f"No fullinertia in {xml_path}")
    vals = tuple(float(v) for v in full.split())
    if len(vals) != 6:
        raise ValueError(f"Expected 6 full inertia values in {xml_path}, got {vals}")
    return vals


def _mesh_bounds_obj(
    mesh_path: Path,
) -> tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]]:
    mins = np.array([np.inf, np.inf, np.inf], dtype=float)
    maxs = np.array([-np.inf, -np.inf, -np.inf], dtype=float)
    with open(mesh_path, "r", errors="ignore") as f:
        for line in f:
            if not line.startswith("v "):
                continue
            toks = line.split()
            if len(toks) < 4:
                continue
            v = np.array([float(toks[1]), float(toks[2]), float(toks[3])], dtype=float)
            mins = np.minimum(mins, v)
            maxs = np.maximum(maxs, v)
    if not np.isfinite(mins).all() or not np.isfinite(maxs).all():
        raise ValueError(f"Could not read vertices from mesh: {mesh_path}")
    extents = maxs - mins
    return (
        (float(mins[0]), float(mins[1]), float(mins[2])),
        (float(maxs[0]), float(maxs[1]), float(maxs[2])),
        (float(extents[0]), float(extents[1]), float(extents[2])),
    )


def _sanitize_friction(raw: list[float] | tuple[float, ...] | np.ndarray) -> tuple[float, float, float]:
    vals = np.asarray(raw, dtype=float)
    if vals.shape[0] < 3:
        vals = np.array([0.9, 0.005, 0.0001], dtype=float)
    vals = vals[:3]
    vals[0] = float(np.clip(vals[0], 0.3, 2.5))
    vals[1] = float(np.clip(vals[1], 0.001, 0.05))
    vals[2] = float(np.clip(vals[2], 0.00005, 0.01))
    return float(vals[0]), float(vals[1]), float(vals[2])


def _pick_grasp_profile(object_id: str, extents: tuple[float, float, float]) -> GraspProfile:
    if object_id in _GRASP_OVERRIDES:
        return _GRASP_PROFILES[_GRASP_OVERRIDES[object_id]]

    ex, ey, ez = extents
    max_dim = max(extents)

    if ez > 0.16 and max(ex, ey) < 0.11:
        return _GRASP_PROFILES["tall"]
    if ez < 0.04:
        return _GRASP_PROFILES["flat"]
    if max_dim < 0.06:
        return _GRASP_PROFILES["small"]
    return _GRASP_PROFILES["default"]


def _audit_spec(
    mass_kg: float,
    inertia_diag: tuple[float, float, float],
    full_inertia: tuple[float, float, float, float, float, float],
    friction: tuple[float, float, float],
    extents_m: tuple[float, float, float],
) -> tuple[str, ...]:
    warnings: list[str] = []

    if mass_kg <= 0:
        warnings.append("non_positive_mass")
    if mass_kg < 0.001:
        warnings.append("very_low_mass")
    if mass_kg > 6.0:
        warnings.append("very_high_mass")

    ixx, iyy, izz = inertia_diag
    if ixx <= 0 or iyy <= 0 or izz <= 0:
        warnings.append("non_positive_inertia_diag")

    # Rigid-body principal moments should satisfy triangle inequalities.
    if ixx + iyy < izz or ixx + izz < iyy or iyy + izz < ixx:
        warnings.append("inertia_triangle_inequality_violation")

    if friction[0] < 0.2 or friction[0] > 3.0:
        warnings.append("friction_slide_out_of_range")

    vol_box = max(extents_m[0] * extents_m[1] * extents_m[2], 1e-8)
    density = mass_kg / vol_box
    if density < 20:
        warnings.append("very_low_box_density")
    if density > 40000:
        warnings.append("very_high_box_density")

    if not np.isfinite(np.asarray(full_inertia)).all():
        warnings.append("non_finite_full_inertia")

    return tuple(warnings)


def build_object_spec(object_id: str, ycb_base: Path = DEFAULT_YCB_BASE) -> YCBObjectSpec:
    json_path = ycb_base / object_id / f"{object_id}_physics.json"
    xml_path = ycb_base / object_id / f"{object_id}.xml"
    if not json_path.exists():
        raise FileNotFoundError(json_path)
    if not xml_path.exists():
        raise FileNotFoundError(xml_path)

    phys = json.loads(json_path.read_text())
    mass_kg = float(phys["mass_kg"])
    com_m = tuple(float(v) for v in phys["com_m"])
    inertia_diag = tuple(float(v) for v in phys["inertia_diag"])
    full_inertia = _read_full_inertia(xml_path)

    mesh_visual = str(Path(phys["mesh_visual"]).resolve())
    mesh_collision = str(Path(phys["mesh_collision"]).resolve())
    bounds_min_m, bounds_max_m, extents_m = _mesh_bounds_obj(Path(mesh_collision))

    if object_id in _CONTACT_OVERRIDES:
        contact = _CONTACT_OVERRIDES[object_id]
        friction = contact.friction
    else:
        friction = _sanitize_friction(phys.get("friction", [0.9, 0.005, 0.0001]))
        contact = ContactParams(friction=friction)

    grasp = _pick_grasp_profile(object_id, extents_m)

    audit_warnings = _audit_spec(
        mass_kg=mass_kg,
        inertia_diag=inertia_diag,
        full_inertia=full_inertia,
        friction=friction,
        extents_m=extents_m,
    )

    return YCBObjectSpec(
        object_id=object_id,
        mass_kg=mass_kg,
        com_m=com_m,
        inertia_diag=inertia_diag,
        full_inertia=full_inertia,
        mesh_visual=mesh_visual,
        mesh_collision=mesh_collision,
        bounds_min_m=bounds_min_m,
        bounds_max_m=bounds_max_m,
        extents_m=extents_m,
        contact=contact,
        grasp=grasp,
        audit_warnings=audit_warnings,
    )


def build_manipulation_library(
    ycb_base: Path = DEFAULT_YCB_BASE,
    object_ids: list[str] | None = None,
) -> list[YCBObjectSpec]:
    ids = object_ids if object_ids is not None else discover_object_ids(ycb_base)
    return [build_object_spec(object_id=obj_id, ycb_base=ycb_base) for obj_id in ids]


def write_library_json(output_path: Path, specs: list[YCBObjectSpec]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "ycb_base": str(DEFAULT_YCB_BASE),
        "num_objects": len(specs),
        "objects": [asdict(spec) for spec in specs],
    }
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n")
