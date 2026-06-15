import argparse
import math
import re
from pathlib import Path
from textwrap import dedent
from typing import Iterable, Optional


def safe_name(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"[^a-z0-9_\-]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    if not s:
        raise ValueError("Empty object name after sanitization.")
    return s


def write_text(path: Path, content: str, overwrite: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing file: {path}")
    path.write_text(content, encoding="utf-8")


def copy_file(src: Path, dst: Path, overwrite: bool) -> bool:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() and not overwrite:
        return False
    dst.write_bytes(src.read_bytes())
    return True


def copy_all_files(src_dir: Path, dst_dir: Path, overwrite: bool) -> tuple[int, int]:
    files = sorted([p for p in src_dir.rglob("*") if p.is_file()])
    if not files:
        raise FileNotFoundError(f"No source files found in: {src_dir}")

    copied = 0
    for src in files:
        rel = src.relative_to(src_dir)
        dst = dst_dir / rel
        did_copy = copy_file(src, dst, overwrite=overwrite)
        copied += int(did_copy)
    return len(files), copied


def resolve_source_dir(raw_source: Path) -> Path:
    """Resolve shorthand folders (e.g. ycb/bowl -> ycb/024_bowl/google_16k)."""
    if raw_source.exists():
        source_dir = raw_source if raw_source.is_dir() else raw_source.parent
    else:
        parent = raw_source.parent if raw_source.parent != Path("") else Path(".")
        if not parent.exists():
            raise FileNotFoundError(f"Parent directory not found: {parent}")
        token = raw_source.name.lower()
        candidates = sorted(
            [
                p
                for p in parent.iterdir()
                if p.is_dir()
                and (
                    p.name.lower() == token
                    or p.name.lower().endswith(f"_{token}")
                    or token in p.name.lower()
                )
            ]
        )
        if not candidates:
            raise FileNotFoundError(
                f"Could not resolve source directory from: {raw_source}\n"
                f"No matching directory found under: {parent}"
            )
        source_dir = candidates[0]

    for subdir in ("google_16k", "google_512k"):
        candidate = source_dir / subdir
        if candidate.is_dir():
            return candidate
    return source_dir


def pick_file(
    source_dir: Path,
    suffix: str,
    preferred_names: Iterable[str],
    required: bool = True,
) -> Optional[Path]:
    suffix = suffix.lower()
    candidates = sorted(
        [p for p in source_dir.rglob("*") if p.is_file() and p.suffix.lower() == suffix],
        key=lambda p: p.as_posix(),
    )

    preferred_lower = [name.lower() for name in preferred_names if name]
    for preferred in preferred_lower:
        for candidate in candidates:
            if candidate.name.lower() == preferred:
                return candidate

    if candidates:
        return candidates[0]

    if required:
        raise FileNotFoundError(f"No '*{suffix}' file found under: {source_dir}")
    return None


def compute_obj_bbox(obj_path: Path) -> tuple[list[float], list[float]]:
    mins = [float("inf"), float("inf"), float("inf")]
    maxs = [float("-inf"), float("-inf"), float("-inf")]
    num_vertices = 0

    for line in obj_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if not line.startswith("v "):
            continue
        parts = line.split()
        if len(parts) < 4:
            continue
        coords = [float(parts[1]), float(parts[2]), float(parts[3])]
        for axis, value in enumerate(coords):
            mins[axis] = min(mins[axis], value)
            maxs[axis] = max(maxs[axis], value)
        num_vertices += 1

    if num_vertices == 0:
        raise ValueError(f"OBJ file contains no vertices: {obj_path}")

    return mins, maxs


def parse_floats(arg_name: str, raw_value: str, expected_len: int) -> str:
    parts = raw_value.strip().split()
    if len(parts) != expected_len:
        raise ValueError(f"{arg_name} must have exactly {expected_len} floats, got: {raw_value!r}")
    try:
        values = [float(part) for part in parts]
    except ValueError as exc:
        raise ValueError(f"{arg_name} must contain only floats, got: {raw_value!r}") from exc
    return " ".join(f"{value:g}" for value in values)


def parse_scalar(arg_name: str, raw_value: str) -> str:
    try:
        value = float(raw_value)
    except ValueError as exc:
        raise ValueError(f"{arg_name} must be a float, got: {raw_value!r}") from exc
    return f"{value:g}"


def main():
    ap = argparse.ArgumentParser(
        description="Generate benchmark-style dual XML assets from a YCB object directory."
    )
    ap.add_argument("--root_dir", required=True)
    ap.add_argument(
        "--source_dir",
        type=str,
        default=None,
        help="Object directory (supports shorthand like './ycb/bowl').",
    )
    ap.add_argument(
        "--obj",
        type=str,
        default=None,
        help="Optional visual OBJ path (e.g. './ycb/024_bowl/google_16k/textured.obj').",
    )
    ap.add_argument("--name", required=True)
    ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--density", type=float, default=300.0)
    ap.add_argument("--friction", type=str, default="1 0.3 0.1")
    ap.add_argument(
        "--bowl_split_collision",
        action="store_true",
        help="For bowl-like objects, split collision into smooth bottom + rough sidewall ring.",
    )
    ap.add_argument("--bowl_bottom_friction", type=str, default="0.2 0.02 0.01")
    ap.add_argument("--bowl_wall_friction", type=str, default="2.2 0.15 0.05")
    ap.add_argument("--bowl_bottom_radius", type=float, default=0.055)
    ap.add_argument("--bowl_bottom_thickness", type=float, default=0.003)
    ap.add_argument("--bowl_wall_radius", type=float, default=0.078)
    ap.add_argument("--bowl_wall_height", type=float, default=0.032)
    ap.add_argument("--bowl_wall_capsule_radius", type=float, default=0.006)
    ap.add_argument("--bowl_wall_capsule_half_length", type=float, default=0.020)
    ap.add_argument("--bowl_wall_segments", type=int, default=12)
    ap.add_argument("--object_pos", type=str, default="0.3 0 0.02")
    ap.add_argument("--target_pos", type=str, default="0.45 0 0.02")
    ap.add_argument("--target_rgba", type=str, default="0.5 0.5 0.5 0.2")
    ap.add_argument("--site_size", type=str, default="0.005")
    ap.add_argument(
        "--add_grasp_handles",
        action="store_true",
        help="Add extra collision grasp handles (disabled by default).",
    )
    ap.add_argument("--grasp_handle_offset", type=float, default=0.065)
    ap.add_argument("--grasp_handle_z", type=float, default=0.032)
    ap.add_argument("--grasp_handle_size", type=str, default="0.014 0.008 0.012")
    ap.add_argument("--grasp_handle_friction", type=str, default="2.5 0.2 0.05")
    ap.add_argument("--grasp_handle_mass", type=float, default=0.0)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    root_dir = Path(args.root_dir).resolve()
    obj_name = safe_name(args.name)

    raw_source_dir = None
    if args.source_dir is not None:
        raw_source_dir = Path(args.source_dir).expanduser()
    elif args.obj is not None:
        raw_source_dir = Path(args.obj).expanduser().parent
    else:
        raise ValueError("Either --source_dir or --obj must be provided.")

    src_dir = resolve_source_dir(raw_source_dir)

    obj_hint = Path(args.obj).name if args.obj else ""
    obj_path = pick_file(src_dir, suffix=".obj", preferred_names=[obj_hint, "textured.obj"], required=True)
    stl_path = pick_file(src_dir, suffix=".stl", preferred_names=["nontextured.stl"], required=True)
    png_path = pick_file(src_dir, suffix=".png", preferred_names=["texture_map.png"], required=True)
    mtl_path = pick_file(src_dir, suffix=".mtl", preferred_names=["textured.mtl"], required=False)

    assets_root = root_dir / "assets"
    assets_dir = assets_root / obj_name
    assets_dir.mkdir(parents=True, exist_ok=True)
    total_files, copied_files = copy_all_files(src_dir, assets_dir, overwrite=args.overwrite)

    scale = args.scale
    scale_str = f"{scale:g} {scale:g} {scale:g}"
    friction_str = parse_floats("--friction", args.friction, expected_len=3)
    bowl_bottom_friction = parse_floats("--bowl_bottom_friction", args.bowl_bottom_friction, expected_len=3)
    bowl_wall_friction = parse_floats("--bowl_wall_friction", args.bowl_wall_friction, expected_len=3)
    object_pos = parse_floats("--object_pos", args.object_pos, expected_len=3)
    target_pos = parse_floats("--target_pos", args.target_pos, expected_len=3)
    target_rgba = parse_floats("--target_rgba", args.target_rgba, expected_len=4)
    site_size = parse_scalar("--site_size", args.site_size)
    grasp_handle_size = parse_floats("--grasp_handle_size", args.grasp_handle_size, expected_len=3)
    grasp_handle_friction = parse_floats("--grasp_handle_friction", args.grasp_handle_friction, expected_len=3)
    grasp_handle_offset = f"{float(args.grasp_handle_offset):g}"
    grasp_handle_z = f"{float(args.grasp_handle_z):g}"
    grasp_handle_mass = f"{float(args.grasp_handle_mass):g}"
    add_grasp_handles = args.add_grasp_handles
    is_bowl_like = "bowl" in obj_name
    use_split_bowl_collision = is_bowl_like and bool(args.bowl_split_collision)
    bowl_wall_segments = int(args.bowl_wall_segments)
    if bowl_wall_segments < 3:
        raise ValueError("--bowl_wall_segments must be >= 3.")

    vis_mesh_name = f"{obj_name}_vis_mesh"
    col_mesh_name = f"{obj_name}_col_mesh"
    tex_name = f"tex-{obj_name}"
    mat_name = f"{obj_name}_material"
    collision_class = f"{obj_name}_collision"
    collision_mass_class = f"{obj_name}_collision_mass"
    collision_bottom_class = f"{obj_name}_collision_bottom"
    collision_wall_class = f"{obj_name}_collision_wall"
    visual_class = f"{obj_name}_visual"
    target_class = f"{obj_name}_target"
    site_class = f"{obj_name}_com"
    grasp_handle_class = f"{obj_name}_grasp_handle"

    obj_rel = obj_path.relative_to(src_dir).as_posix()
    stl_rel = stl_path.relative_to(src_dir).as_posix()
    png_rel = png_path.relative_to(src_dir).as_posix()

    collision_solimp = "0.95 0.99 0.001" if is_bowl_like else "0.998 0.998 0.001"
    collision_solref = "0.004 1" if is_bowl_like else "0.001 1"

    mesh_geom_pos = "0 0 0"
    mesh_recenter_summary = None
    if is_bowl_like:
        obj_bbox_mins, obj_bbox_maxs = compute_obj_bbox(obj_path)
        mesh_center_x = 0.5 * (obj_bbox_mins[0] + obj_bbox_maxs[0])
        mesh_center_y = 0.5 * (obj_bbox_mins[1] + obj_bbox_maxs[1])
        mesh_bottom_z = obj_bbox_mins[2]
        mesh_recenter_offset = (-mesh_center_x, -mesh_center_y, -mesh_bottom_z)
        mesh_geom_pos = " ".join(f"{value:g}" for value in mesh_recenter_offset)
        mesh_recenter_summary = (
            f"bbox center=({mesh_center_x:g}, {mesh_center_y:g}), "
            f"bottom_z={mesh_bottom_z:g}, applied pos=({mesh_geom_pos})"
        )

    bowl_bottom_radius = args.bowl_bottom_radius * scale
    bowl_bottom_thickness = args.bowl_bottom_thickness * scale
    bowl_wall_radius = args.bowl_wall_radius * scale
    bowl_wall_height = args.bowl_wall_height * scale
    bowl_wall_capsule_radius = args.bowl_wall_capsule_radius * scale
    bowl_wall_capsule_half_length = args.bowl_wall_capsule_half_length * scale
    bowl_bottom_size = f"{bowl_bottom_radius:g} {bowl_bottom_thickness:g}"
    bowl_wall_size = f"{bowl_wall_capsule_radius:g} {bowl_wall_capsule_half_length:g}"

    grasp_handle_default = ""
    if add_grasp_handles:
        grasp_handle_default = dedent(
            f"""\
                <default class="{grasp_handle_class}">
                  <geom type="box"
                        size="{grasp_handle_size}"
                        mass="{grasp_handle_mass}"
                        friction="{grasp_handle_friction}"
                        solimp="0.92 0.98 0.001"
                        solref="0.01 1"
                        margin="0.001"
                        priority="1"
                        contype="1"
                        conaffinity="1"
                        group="1"/>
                </default>
            """
        )

    collision_default_block = dedent(
        f"""\
            <default class="{collision_class}">
              <geom type="mesh"
                    mesh="{col_mesh_name}"
                    pos="{mesh_geom_pos}"
                    density="{args.density:g}"
                    friction="{friction_str}"
                    solimp="{collision_solimp}"
                    solref="{collision_solref}"
                    rgba="0 0 0 0"
                    contype="1"
                    conaffinity="1"
                    group="3"/>
            </default>
        """
    )
    if use_split_bowl_collision:
        collision_default_block = dedent(
            f"""\
                <default class="{collision_mass_class}">
                  <geom type="mesh"
                        mesh="{col_mesh_name}"
                        pos="{mesh_geom_pos}"
                        density="{args.density:g}"
                        rgba="0 0 0 0"
                        contype="0"
                        conaffinity="0"
                        group="3"/>
                </default>
                <default class="{collision_bottom_class}">
                  <geom type="cylinder"
                        size="{bowl_bottom_size}"
                        mass="0"
                        friction="{bowl_bottom_friction}"
                        solimp="{collision_solimp}"
                        solref="{collision_solref}"
                        rgba="0 0 0 0"
                        contype="1"
                        conaffinity="1"
                        group="3"/>
                </default>
                <default class="{collision_wall_class}">
                  <geom type="capsule"
                        size="{bowl_wall_size}"
                        mass="0"
                        friction="{bowl_wall_friction}"
                        solimp="{collision_solimp}"
                        solref="{collision_solref}"
                        rgba="0 0 0 0"
                        contype="1"
                        conaffinity="1"
                        group="3"/>
                </default>
            """
        )

    outer_content = dedent(
        f"""\
        <mujoco model="{obj_name}_outer">
          <asset>
            <mesh name="{vis_mesh_name}" file="{obj_rel}" scale="{scale_str}"/>
            <mesh name="{col_mesh_name}" file="{stl_rel}" scale="{scale_str}"/>
            <texture name="{tex_name}" type="2d" file="{png_rel}"/>
            <material name="{mat_name}"
                      texture="{tex_name}"
                      rgba="1 1 1 1"
                      texrepeat="1 1"
                      texuniform="true"
                      reflectance="0.5"/>
          </asset>

          <default>
{collision_default_block.rstrip()}
            <default class="{visual_class}">
              <geom type="mesh"
                    mesh="{vis_mesh_name}"
                    pos="{mesh_geom_pos}"
                    material="{mat_name}"
                    contype="0"
                    conaffinity="0"
                    group="0"/>
            </default>
            <default class="{target_class}">
              <geom type="mesh"
                    mesh="{vis_mesh_name}"
                    pos="{mesh_geom_pos}"
                    material="{mat_name}"
                    rgba="{target_rgba}"
                    contype="0"
                    conaffinity="0"
                    group="0"/>
            </default>
            <default class="{site_class}">
              <site type="sphere" size="{site_size}" rgba="0 1 0 1" group="5"/>
            </default>
{grasp_handle_default.rstrip()}
          </default>

          <worldbody>
          </worldbody>
        </mujoco>
        """
    )

    grasp_handle_geoms = ""
    if add_grasp_handles:
        grasp_handle_geoms = dedent(
            f"""\
                  <geom name="object_0_grasp_handle_pos"
                        class="{grasp_handle_class}"
                        pos="{grasp_handle_offset} 0 {grasp_handle_z}"/>
                  <geom name="object_0_grasp_handle_neg"
                        class="{grasp_handle_class}"
                        pos="-{grasp_handle_offset} 0 {grasp_handle_z}"/>
            """
        )

    split_collision_geoms = ""
    if use_split_bowl_collision:
        wall_geom_lines = []
        for i in range(bowl_wall_segments):
            theta = 2.0 * math.pi * i / bowl_wall_segments
            cx = bowl_wall_radius * math.cos(theta)
            cy = bowl_wall_radius * math.sin(theta)
            tx = -math.sin(theta)
            ty = math.cos(theta)
            x1 = cx - tx * bowl_wall_capsule_half_length
            y1 = cy - ty * bowl_wall_capsule_half_length
            x2 = cx + tx * bowl_wall_capsule_half_length
            y2 = cy + ty * bowl_wall_capsule_half_length
            wall_geom_lines.append(
                f'              <geom name="object_0_wall_{i}" class="{collision_wall_class}" '
                f'fromto="{x1:g} {y1:g} {bowl_wall_height:g} {x2:g} {y2:g} {bowl_wall_height:g}"/>'
            )
        split_collision_geoms = "\n".join(
            [
                f'              <geom name="object_0_mass" class="{collision_mass_class}"/>',
                f'              <geom name="object_0_bottom" class="{collision_bottom_class}" '
                f'pos="0 0 {bowl_bottom_thickness:g}"/>',
                *wall_geom_lines,
            ]
        )

    inner_content = dedent(
        f"""\
        <mujoco model="{obj_name}_inner">
          <worldbody>
            <body name="object_0" pos="{object_pos}">
              <freejoint name="object_joint_0"/>
{split_collision_geoms if use_split_bowl_collision else f'              <geom name="object_0_collision" class="{collision_class}"/>'}
              <geom name="object_0_visual" class="{visual_class}"/>
              <site name="com_0" class="{site_class}"/>
{grasp_handle_geoms.rstrip()}
            </body>
            <body name="object_target_0" pos="{target_pos}" mocap="true">
              <geom name="target_object_0" class="{target_class}"/>
            </body>
          </worldbody>
        </mujoco>
        """
    )

    dst_inner = assets_dir / "inner.xml"
    dst_outer = assets_dir / "outer.xml"
    write_text(dst_inner, inner_content, overwrite=args.overwrite)
    write_text(dst_outer, outer_content, overwrite=args.overwrite)

    print("Generated benchmark-style dual XML assets:")
    print(f"  ROOT  : {root_dir}")
    print(f"  SRC   : {src_dir}")
    print(f"  FILES : copied {copied_files}/{total_files} files into {assets_dir}")
    print(f"  INNER : {dst_inner}")
    print(f"  OUTER : {dst_outer}")
    print(f"  OBJ   : {(assets_dir / obj_rel)}")
    print(f"  GRASP_HANDLES : {'enabled' if add_grasp_handles else 'disabled'}")
    if mtl_path is not None:
        print(f"  MTL   : {(assets_dir / mtl_path.relative_to(src_dir))} (copied if needed)")
    else:
        print("  MTL   : NOT found (optional)")
    print(f"  PNG   : {(assets_dir / png_rel)}")
    print(f"  STL   : {(assets_dir / stl_rel)}")
    if mesh_recenter_summary is not None:
        print(f"  MESH_RECENTER : {mesh_recenter_summary}")
    if use_split_bowl_collision:
        print(
            "  BOWL_COLLISION : split (bottom friction="
            f"{bowl_bottom_friction}, wall friction={bowl_wall_friction}, segments={bowl_wall_segments})"
        )
    else:
        print(f"  BOWL_COLLISION : mesh (friction={friction_str})")


if __name__ == "__main__":
    main()
