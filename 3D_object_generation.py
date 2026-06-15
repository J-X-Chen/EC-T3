import argparse
import colorsys
from io import BytesIO
import shutil
import subprocess
from pathlib import Path
from textwrap import dedent

import numpy as np
from PIL import Image, ImageDraw


import json
import re
import struct


try:
    import coacd
except ImportError as exc:  # pragma: no cover - runtime environment dependent
    raise ImportError(
        "coacd is not available in the current Python environment. "
        "Run this script inside the conda env where coacd is installed, "
        "for example: `conda run -n stanet python xxx.py ...`"
    ) from exc


SIZE_PRESETS = {
    "very_small": 0.01,
    "small": 0.05,
    "middle": 0.07,
    "large": 0.12,
    "very_large": 0.2,
}


GLB_JSON_CHUNK = 0x4E4F534A
GLB_BIN_CHUNK = 0x004E4942
DEFAULT_GLB_OBJECT_QUAT = [0.70710678, 0.70710678, 0.0, 0.0]
ASSET_GLB_OBJECT_QUATS = {
    "knife": [0.70710678, 0.0, 0.70710678, 0.0],
}
DEFAULT_GEOM_Z_OFFSET = 0.03
DEFAULT_BOWL_GEOM_Z_OFFSET = 0.05
YCB_SITE_PROFILES = {
    "bowl": {"bottom_z": -0.03, "top_z": 0.03, "horizontal_radius": 0.06},
    "small_basin": {"bottom_z": -0.03, "top_z": 0.03, "horizontal_radius": 0.06},
    "fork": {"bottom_z": -0.01, "top_z": 0.01, "horizontal_radius": 0.08},
    "knife": {"bottom_z": -0.01, "top_z": 0.01, "horizontal_radius": 0.11},
    "mug": {"bottom_z": -0.05, "top_z": 0.07, "horizontal_radius": 0.05},
    "cup": {"bottom_z": -0.05, "top_z": 0.07, "horizontal_radius": 0.05},
    "plate": {"bottom_z": -0.01, "top_z": 0.01, "horizontal_radius": 0.11},
    "spoon": {"bottom_z": -0.012, "top_z": 0.012, "horizontal_radius": 0.09},
}

COMPONENT_DTYPES = {
    5120: np.int8,
    5121: np.uint8,
    5122: np.int16,
    5123: np.uint16,
    5125: np.uint32,
    5126: np.float32,
}

NUM_COMPONENTS = {
    "SCALAR": 1,
    "VEC2": 2,
    "VEC3": 3,
    "VEC4": 4,
    "MAT2": 4,
    "MAT3": 9,
    "MAT4": 16,
}


def safe_name(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"[^a-z0-9_\-]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    if not s:
        raise ValueError("Empty object name after sanitization.")
    return s


def normalize_wxyz_quat(quat) -> np.ndarray:
    quat = np.asarray(quat, dtype=np.float64)
    norm = np.linalg.norm(quat)
    if norm == 0:
        raise ValueError("Quaternion must be non-zero.")
    return quat / norm


def get_ycb_site_profile(asset_name: str):
    return YCB_SITE_PROFILES.get(asset_name)


def default_geom_z_offset_for_asset(asset_name: str) -> float:
    return DEFAULT_BOWL_GEOM_Z_OFFSET if asset_name == "bowl" else DEFAULT_GEOM_Z_OFFSET


def default_glb_object_quat_for_asset(asset_name: str) -> list[float]:
    return ASSET_GLB_OBJECT_QUATS.get(asset_name, DEFAULT_GLB_OBJECT_QUAT)


def split_numbered_name(name: str) -> tuple[str, str]:
    safe = safe_name(name)
    match = re.fullmatch(r"(.+?)(\d+)", safe)
    if match is None:
        raise ValueError(
            f"Expected --name to be in the form <object><index>, for example bowl01. Got: {name!r}"
        )
    asset_name, variant_id = match.groups()
    return asset_name.rstrip("_-"), variant_id


def write_text(path: Path, content: str, overwrite: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing file: {path}")
    path.write_text(content, encoding="utf-8")


def write_bytes(path: Path, data: bytes, overwrite: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing file: {path}")
    path.write_bytes(data)


def insert_before_closing_tag(path: Path, closing_tag: str, snippet: str) -> bool:
    text = path.read_text(encoding="utf-8")
    if snippet.strip() in text:
        return False
    marker = text.rfind(closing_tag)
    if marker == -1:
        raise ValueError(f"Closing tag {closing_tag!r} not found in {path}")
    insert = snippet
    if marker > 0 and text[marker - 1] != "\n":
        insert = "\n" + insert
    updated = text[:marker] + insert + text[marker:]
    path.write_text(updated, encoding="utf-8")
    return True


def upsert_include_line(path: Path, include_line: str) -> str:
    text = path.read_text(encoding="utf-8")
    match = re.search(r'file="([^"]+)"', include_line)
    if match is None:
        raise ValueError(f"Could not parse include file from line: {include_line}")
    include_file = re.escape(match.group(1))
    include_pattern = re.compile(rf'^[ \t]*<include\s+file="{include_file}"\s*/>\s*$', re.MULTILINE)
    if include_pattern.search(text):
        return "already matched"
    insert_before_closing_tag(path, "</mujoco>", f"  {include_line}\n")
    return "inserted"


def upsert_body_block(path: Path, object_name: str, layout_snippet: str) -> str:
    text = path.read_text(encoding="utf-8")
    body_pattern = re.compile(
        rf'^[ \t]*<body\s+name="{re.escape(object_name)}"[^>]*>\s*.*?^[ \t]*</body>\s*$',
        re.MULTILINE | re.DOTALL,
    )
    match = body_pattern.search(text)
    if match:
        existing = match.group(0).strip()
        replacement = layout_snippet.rstrip()
        if existing == replacement:
            return "already matched"
        updated = text[: match.start()] + replacement + "\n" + text[match.end() :]
        path.write_text(updated, encoding="utf-8")
        return "replaced"
    insert_before_closing_tag(path, "  </worldbody>", layout_snippet)
    return "inserted"


def update_scene_files(defs_path: Path, layout_path: Path, include_line: str, object_name: str, layout_snippet: str) -> tuple[str, str]:
    defs_status = upsert_include_line(defs_path, include_line)
    layout_status = upsert_body_block(layout_path, object_name, layout_snippet)
    return defs_status, layout_status


def find_glb_for_asset(asset_dir: Path) -> Path:
    glbs = sorted(asset_dir.rglob("*.glb"))
    if not glbs:
        raise FileNotFoundError(f"No .glb file found under {asset_dir}")
    return max(glbs, key=lambda path: (path.stat().st_mtime, path.as_posix()))


def parse_glb(glb_path: Path):
    data = glb_path.read_bytes()
    magic, version, length = struct.unpack_from("<4sII", data, 0)
    if magic != b"glTF":
        raise ValueError(f"{glb_path} is not a valid GLB file")
    if version != 2:
        raise ValueError(f"Only GLB 2.0 is supported, got version {version}")

    offset = 12
    json_chunk = None
    bin_chunk = None
    while offset < length:
        chunk_len, chunk_type = struct.unpack_from("<II", data, offset)
        offset += 8
        chunk_data = data[offset : offset + chunk_len]
        offset += chunk_len
        if chunk_type == GLB_JSON_CHUNK:
            json_chunk = chunk_data
        elif chunk_type == GLB_BIN_CHUNK:
            bin_chunk = chunk_data

    if json_chunk is None or bin_chunk is None:
        raise ValueError(f"{glb_path} is missing required JSON or BIN chunks")

    gltf = json.loads(json_chunk.decode("utf-8"))
    return gltf, bin_chunk


def accessor_array(gltf, bin_chunk: bytes, accessor_index: int):
    accessor = gltf["accessors"][accessor_index]
    view = gltf["bufferViews"][accessor["bufferView"]]
    dtype = COMPONENT_DTYPES[accessor["componentType"]]
    component_count = NUM_COMPONENTS[accessor["type"]]
    byte_offset = view.get("byteOffset", 0) + accessor.get("byteOffset", 0)
    count = accessor["count"]
    arr = np.frombuffer(bin_chunk, dtype=dtype, count=count * component_count, offset=byte_offset)
    return arr.reshape(count, component_count)


def node_matrix(node):
    if "matrix" in node:
        return np.array(node["matrix"], dtype=np.float64).reshape(4, 4).T

    translation = np.array(node.get("translation", [0.0, 0.0, 0.0]), dtype=np.float64)
    scale = np.array(node.get("scale", [1.0, 1.0, 1.0]), dtype=np.float64)
    qx, qy, qz, qw = node.get("rotation", [0.0, 0.0, 0.0, 1.0])
    x2, y2, z2 = qx + qx, qy + qy, qz + qz
    xx, yy, zz = qx * x2, qy * y2, qz * z2
    xy, xz, yz = qx * y2, qx * z2, qy * z2
    wx, wy, wz = qw * x2, qw * y2, qw * z2
    rot = np.array(
        [
            [1.0 - (yy + zz), xy - wz, xz + wy, 0.0],
            [xy + wz, 1.0 - (xx + zz), yz - wx, 0.0],
            [xz - wy, yz + wx, 1.0 - (xx + yy), 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    scl = np.diag([scale[0], scale[1], scale[2], 1.0])
    trs = np.eye(4, dtype=np.float64)
    trs[:3, 3] = translation
    return trs @ rot @ scl


def scene_root_nodes(gltf):
    scene_index = gltf.get("scene", 0)
    return gltf["scenes"][scene_index]["nodes"]


def collect_primitives(gltf, bin_chunk: bytes):
    primitives = []

    def walk(node_index: int, parent_matrix: np.ndarray):
        node = gltf["nodes"][node_index]
        transform = parent_matrix @ node_matrix(node)
        if "mesh" in node:
            mesh = gltf["meshes"][node["mesh"]]
            normal_transform = np.linalg.inv(transform[:3, :3]).T
            for prim in mesh["primitives"]:
                positions = accessor_array(gltf, bin_chunk, prim["attributes"]["POSITION"]).astype(np.float64)
                positions_h = np.c_[positions, np.ones(len(positions))]
                positions = (positions_h @ transform.T)[:, :3]

                texcoords = None
                if "TEXCOORD_0" in prim["attributes"]:
                    texcoords = accessor_array(gltf, bin_chunk, prim["attributes"]["TEXCOORD_0"]).astype(np.float64)

                normals = None
                if "NORMAL" in prim["attributes"]:
                    normals = accessor_array(gltf, bin_chunk, prim["attributes"]["NORMAL"]).astype(np.float64)
                    normals = normals @ normal_transform.T
                    norms = np.linalg.norm(normals, axis=1, keepdims=True)
                    normalized = np.zeros_like(normals)
                    normals = np.divide(normals, norms, out=normalized, where=norms > 0)

                if "indices" in prim:
                    indices = accessor_array(gltf, bin_chunk, prim["indices"]).reshape(-1).astype(np.int64)
                else:
                    indices = np.arange(len(positions), dtype=np.int64)

                if prim.get("mode", 4) != 4:
                    raise ValueError("Only triangle primitives (mode=4) are supported")

                primitives.append(
                    {
                        "positions": positions,
                        "texcoords": texcoords,
                        "normals": normals,
                        "indices": indices.reshape(-1, 3),
                        "material": prim.get("material", 0),
                    }
                )
        for child_index in node.get("children", []):
            walk(child_index, transform)

    for root_node in scene_root_nodes(gltf):
        walk(root_node, np.eye(4, dtype=np.float64))
    return primitives


def extract_base_color_texture(gltf, bin_chunk: bytes):
    materials = gltf.get("materials", [])
    if not materials:
        raise ValueError("GLB has no materials")
    base_color = materials[0].get("pbrMetallicRoughness", {}).get("baseColorTexture")
    if base_color is None:
        raise ValueError("GLB has no baseColorTexture")
    tex_index = base_color["index"]
    source_index = gltf["textures"][tex_index]["source"]
    image = gltf["images"][source_index]

    if "bufferView" in image:
        view = gltf["bufferViews"][image["bufferView"]]
        start = view.get("byteOffset", 0)
        end = start + view["byteLength"]
        mime = image.get("mimeType", "image/png")
        ext = ".png" if "png" in mime else ".jpg"
        return bin_chunk[start:end], ext

    if "uri" in image:
        uri = image["uri"]
        image_path = Path(uri)
        return image_path.read_bytes(), image_path.suffix or ".png"

    raise ValueError("Unsupported image source in GLB")


def resize_texture_bytes(texture_bytes: bytes, texture_ext: str, max_size: int | None) -> bytes:
    if max_size is None or max_size <= 0:
        return texture_bytes

    with Image.open(BytesIO(texture_bytes)) as image:
        if max(image.size) <= max_size:
            return texture_bytes

        resampling = getattr(Image, "Resampling", Image).LANCZOS
        image.thumbnail((max_size, max_size), resampling)

        ext = texture_ext.lower()
        output = BytesIO()
        if ext in {".jpg", ".jpeg"}:
            if image.mode not in {"RGB", "L"}:
                image = image.convert("RGB")
            image.save(output, format="JPEG", quality=90, optimize=True)
        else:
            if image.mode not in {"RGB", "RGBA"}:
                image = image.convert("RGBA")
            image.save(output, format="PNG", optimize=True)
        return output.getvalue()


def write_visual_obj(path: Path, primitives, overwrite: bool, flip_v: bool = True) -> None:
    lines = []
    vertex_offset = 1
    texcoord_offset = 1
    normal_offset = 1
    for prim in primitives:
        positions = prim["positions"]
        texcoords = prim["texcoords"]
        normals = prim["normals"]
        indices = prim["indices"]

        for v in positions:
            lines.append(f"v {v[0]:.8f} {v[1]:.8f} {v[2]:.8f}")
        if texcoords is not None:
            for vt in texcoords:
                v_coord = 1.0 - vt[1] if flip_v else vt[1]
                lines.append(f"vt {vt[0]:.8f} {v_coord:.8f}")
        if normals is not None:
            for vn in normals:
                lines.append(f"vn {vn[0]:.8f} {vn[1]:.8f} {vn[2]:.8f}")

        for tri in indices:
            parts = []
            for idx in tri:
                v = vertex_offset + idx
                if texcoords is not None and normals is not None:
                    parts.append(f"{v}/{texcoord_offset + idx}/{normal_offset + idx}")
                elif texcoords is not None:
                    parts.append(f"{v}/{texcoord_offset + idx}")
                elif normals is not None:
                    parts.append(f"{v}//{normal_offset + idx}")
                else:
                    parts.append(str(v))
            lines.append("f " + " ".join(parts))

        vertex_offset += len(positions)
        if texcoords is not None:
            texcoord_offset += len(texcoords)
        if normals is not None:
            normal_offset += len(normals)

    write_text(path, "\n".join(lines) + "\n", overwrite=overwrite)


def count_primitive_faces(primitives) -> int:
    return int(sum(len(prim["indices"]) for prim in primitives))


def blender_obj_decimate(input_obj: Path, output_obj: Path, max_faces: int, blender_bin: str | None, overwrite: bool) -> bool:
    if blender_bin is None:
        return False
    if output_obj.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing file: {output_obj}")

    script = f"""
import bpy
from pathlib import Path

input_obj = {str(input_obj)!r}
output_obj = {str(output_obj)!r}
max_faces = {int(max_faces)}

bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete()

if hasattr(bpy.ops.wm, 'obj_import'):
    bpy.ops.wm.obj_import(filepath=input_obj)
else:
    bpy.ops.import_scene.obj(filepath=input_obj)

mesh_objects = [obj for obj in bpy.context.scene.objects if obj.type == 'MESH']
if not mesh_objects:
    raise RuntimeError('No mesh objects imported from OBJ')

bpy.ops.object.select_all(action='DESELECT')
for obj in mesh_objects:
    obj.select_set(True)
bpy.context.view_layer.objects.active = mesh_objects[0]
if len(mesh_objects) > 1:
    bpy.ops.object.join()
obj = bpy.context.view_layer.objects.active

face_count = len(obj.data.polygons)
if face_count > max_faces:
    modifier = obj.modifiers.new(name='hecrl_decimate', type='DECIMATE')
    modifier.ratio = max(max_faces / face_count, 0.001)
    if hasattr(modifier, 'use_collapse_triangulate'):
        modifier.use_collapse_triangulate = True
    bpy.ops.object.modifier_apply(modifier=modifier.name)

triangulate = obj.modifiers.new(name='hecrl_triangulate', type='TRIANGULATE')
bpy.ops.object.modifier_apply(modifier=triangulate.name)

Path(output_obj).parent.mkdir(parents=True, exist_ok=True)
if hasattr(bpy.ops.wm, 'obj_export'):
    bpy.ops.wm.obj_export(
        filepath=output_obj,
        export_selected_objects=True,
        export_uv=True,
        export_normals=True,
        export_materials=False,
    )
else:
    bpy.ops.export_scene.obj(
        filepath=output_obj,
        use_selection=True,
        use_uvs=True,
        use_normals=True,
        use_materials=False,
    )
"""
    completed = subprocess.run(
        [blender_bin, "--background", "--factory-startup", "--python-expr", script],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    if completed.returncode != 0:
        print("Warning: Blender visual decimation failed; falling back to full visual OBJ.")
        print(completed.stdout[-4000:])
        return False
    return True


def write_visual_obj_for_runtime(
    path: Path,
    primitives,
    overwrite: bool,
    flip_v: bool,
    max_faces: int | None,
    blender_bin: str | None,
) -> str:
    face_count = count_primitive_faces(primitives)
    if max_faces is None or max_faces <= 0 or face_count <= max_faces:
        write_visual_obj(path, primitives, overwrite=overwrite, flip_v=flip_v)
        return f"full ({face_count} faces)"

    tmp_path = path.with_name(f"{path.stem}.full_tmp{path.suffix}")
    write_visual_obj(tmp_path, primitives, overwrite=True, flip_v=flip_v)
    decimated = blender_obj_decimate(
        input_obj=tmp_path,
        output_obj=path,
        max_faces=max_faces,
        blender_bin=blender_bin,
        overwrite=overwrite,
    )
    try:
        tmp_path.unlink()
    except FileNotFoundError:
        pass

    if decimated:
        return f"decimated ({face_count} -> <= {max_faces} faces)"

    write_visual_obj(path, primitives, overwrite=overwrite, flip_v=flip_v)
    return f"full fallback ({face_count} faces)"


def write_collision_obj(path: Path, primitives, overwrite: bool) -> None:
    lines = []
    vertex_offset = 1
    for prim in primitives:
        positions = prim["positions"]
        indices = prim["indices"]
        for v in positions:
            lines.append(f"v {v[0]:.8f} {v[1]:.8f} {v[2]:.8f}")
        for tri in indices:
            a, b, c = tri + vertex_offset
            lines.append(f"f {a} {b} {c}")
        vertex_offset += len(positions)
    write_text(path, "\n".join(lines) + "\n", overwrite=overwrite)


def compute_bounds(primitives, quat=None):
    positions = np.concatenate([prim["positions"] for prim in primitives], axis=0)
    if quat is not None:
        rotation = quat_wxyz_to_rotmat(quat)
        positions = positions @ rotation.T
    mins = positions.min(axis=0)
    maxs = positions.max(axis=0)
    horizontal_radius = np.linalg.norm(positions[:, :2], axis=1).max()
    return mins, maxs, float(horizontal_radius)

def normalize(vec: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vec)
    if norm == 0:
        raise ValueError("Zero-length vector cannot be normalized")
    return vec / norm


def hull_color(index: int, total: int) -> tuple[int, int, int]:
    if total <= 0:
        total = 1
    hue = (index / total + 0.13) % 1.0
    sat = 0.7
    val = 0.95
    r, g, b = colorsys.hsv_to_rgb(hue, sat, val)
    return int(r * 255), int(g * 255), int(b * 255)


def quat_wxyz_to_rotmat(quat) -> np.ndarray:
    quat = np.asarray(quat, dtype=np.float64)
    quat = quat / np.linalg.norm(quat)
    w, x, y, z = quat
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def render_coacd_mask(
    hulls,
    output_path: Path,
    effective_scale: float,
    image_size: int,
    quat,
) -> None:
    rotation = quat_wxyz_to_rotmat(quat)
    all_vertices = [
        (np.asarray(hull_vertices, dtype=np.float64) * effective_scale) @ rotation.T
        for hull_vertices, _ in hulls
    ]
    stacked = np.concatenate(all_vertices, axis=0)
    mins = stacked.min(axis=0)
    maxs = stacked.max(axis=0)
    center = 0.5 * (mins + maxs)

    camera_dir = normalize(np.array([1.0, -0.85, 0.7], dtype=np.float64))
    up_guess = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    right = normalize(np.cross(up_guess, camera_dir))
    up = normalize(np.cross(camera_dir, right))

    projected_vertices = []
    all_xy = []
    for vertices in all_vertices:
        centered = vertices - center
        x = centered @ right
        y = centered @ up
        z = centered @ camera_dir
        projected = np.column_stack([x, y, z])
        projected_vertices.append(projected)
        all_xy.append(projected[:, :2])

    xy = np.concatenate(all_xy, axis=0)
    xy_mins = xy.min(axis=0)
    xy_maxs = xy.max(axis=0)
    span = np.maximum(xy_maxs - xy_mins, 1e-8)
    canvas_extent = float(image_size - 1)
    fit_scale = 0.88 * canvas_extent / max(span[0], span[1])
    xy_center = 0.5 * (xy_mins + xy_maxs)

    img = Image.new("RGB", (image_size, image_size), (0, 0, 0))
    draw = ImageDraw.Draw(img)

    triangles = []
    for hull_index, ((_, hull_faces), projected) in enumerate(zip(hulls, projected_vertices)):
        color = hull_color(hull_index, len(hulls))
        for tri in np.asarray(hull_faces, dtype=np.int32):
            tri_proj = projected[tri]
            pts = tri_proj[:, :2]
            pts = (pts - xy_center) * fit_scale
            pts[:, 0] += image_size / 2.0
            pts[:, 1] = image_size / 2.0 - pts[:, 1]
            depth = float(np.mean(tri_proj[:, 2]))
            triangles.append((depth, color, pts))

    triangles.sort(key=lambda item: item[0])
    for _, color, pts in triangles:
        draw.polygon([(float(x), float(y)) for x, y in pts], fill=color)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(output_path)


def merge_primitives(primitives):
    vertices = []
    faces = []
    vertex_offset = 0
    for prim in primitives:
        positions = prim["positions"].astype(np.float64)
        indices = prim["indices"].astype(np.int32)
        vertices.append(positions)
        faces.append(indices + vertex_offset)
        vertex_offset += len(positions)
    return np.concatenate(vertices, axis=0), np.concatenate(faces, axis=0)


def run_coacd_decomposition(vertices: np.ndarray, faces: np.ndarray, args):
    mesh = coacd.Mesh(vertices, faces)
    try:
        coacd.set_log_level("warn")
    except Exception:
        pass
    return coacd.run_coacd(
        mesh,
        threshold=args.threshold,
        max_convex_hull=args.max_convex_hull,
        preprocess_mode=args.preprocess_mode,
        preprocess_resolution=args.preprocess_resolution,
        resolution=args.resolution,
        mcts_nodes=args.mcts_nodes,
        mcts_iterations=args.mcts_iterations,
        mcts_max_depth=args.mcts_max_depth,
        pca=args.pca,
        merge=not args.no_merge,
        decimate=args.decimate,
        max_ch_vertex=args.max_ch_vertex,
        extrude=args.extrude,
        extrude_margin=args.extrude_margin,
        apx_mode=args.apx_mode,
        seed=args.seed,
        real_metric=args.real_metric,
    )


def write_collision_obj(path: Path, vertices: np.ndarray, faces: np.ndarray, overwrite: bool) -> None:
    lines = []
    for vertex in vertices:
        lines.append(f"v {vertex[0]:.8f} {vertex[1]:.8f} {vertex[2]:.8f}")
    for tri in faces:
        a, b, c = tri + 1
        lines.append(f"f {a} {b} {c}")
    write_text(path, "\n".join(lines) + "\n", overwrite=overwrite)


def build_multi_collision_layout_snippet(object_name: str, prefix: str, collision_mesh_names: list[str], pos, quat, bottom_z: float, top_z: float, horizontal_radius: float) -> str:
    pos_str = " ".join(f"{v:g}" for v in pos)
    quat_str = " ".join(f"{v:g}" for v in quat)
    lines = [
        f'<body name="{object_name}" pos="{pos_str}" quat="{quat_str}">',
        f'  <freejoint name="{object_name}_freejoint" />',
    ]
    for i, mesh_name in enumerate(collision_mesh_names):
        lines.append(
            f'  <geom name="{object_name}_collision_geom_{i}" class="{prefix}_collision" mesh="{mesh_name}" />'
        )
    lines.extend(
        [
            f'  <geom name="{object_name}_visual_geom" class="{prefix}_visual" />',
            f'  <site name="{object_name}_bottom_site" pos="0 0 {bottom_z:g}" size="0.005" rgba="0 0 0 0" />',
            f'  <site name="{object_name}_top_site" pos="0 0 {top_z:g}" size="0.005" rgba="0 0 0 0" />',
            f'  <site name="{object_name}_horizontal_radius_site" pos="{horizontal_radius:g} 0 0" size="0.005" rgba="0 0 0 0" />',
            "</body>",
            "",
        ]
    )
    return "\n".join(lines)


def build_object_xml(
    model_name: str,
    visual_mesh_name: str,
    collision_mesh_names: list[str],
    texture_name: str,
    material_name: str,
    texture_file_name: str,
    visual_file_name: str,
    collision_file_names: list[str],
    scale_str: str,
    density: float,
    friction: str,
    reflectance: float,
    texrepeat: str,
    bottom_z: float,
    top_z: float,
    horizontal_radius: float,
    geom_quat,
    geom_z_offset: float,
    collision_group: int,
) -> str:
    geom_quat_str = " ".join(f"{v:g}" for v in normalize_wxyz_quat(geom_quat))
    geom_pos_str = f"0 0 {geom_z_offset:g}"
    asset_lines = [
        f'    <mesh name="{visual_mesh_name}" file="{visual_file_name}" scale="{scale_str}"/>',
    ]
    for mesh_name, file_name in zip(collision_mesh_names, collision_file_names):
        asset_lines.append(f'    <mesh name="{mesh_name}" file="{file_name}" scale="{scale_str}"/>')
    asset_lines.extend(
        [
            f'    <texture name="{texture_name}" type="2d" file="{texture_file_name}"/>',
            f'    <material name="{material_name}"',
            f'              texture="{texture_name}"',
            '              rgba="1 1 1 1"',
            f'              texrepeat="{texrepeat}"',
            '              texuniform="false"',
            f'              reflectance="{reflectance:g}"/>',
        ]
    )

    collision_geom_lines = []
    for i, mesh_name in enumerate(collision_mesh_names):
        collision_geom_lines.extend(
            [
                f'        <geom name="g{i}"',
                '              type="mesh"',
                f'              pos="{geom_pos_str}"',
                f'              quat="{geom_quat_str}"',
                f'              mesh="{mesh_name}"',
                f'              density="{density:g}"',
                f'              friction="{friction}"',
                '              rgba="0 0 0 0"',
                '              solimp="0.998 0.998 0.001"',
                '              solref="0.001 1"',
                '              contype="1"',
                '              conaffinity="1"',
                f'              group="{int(collision_group)}"',
                '              condim="4"/>',
            ]
        )

    visual_geom_lines = [
        '        <geom name="g_visual"',
        '              type="mesh"',
        f'              pos="{geom_pos_str}"',
        f'              quat="{geom_quat_str}"',
        f'              mesh="{visual_mesh_name}"',
        f'              material="{material_name}"',
        '              contype="0"',
        '              conaffinity="0"',
        '              group="1"/>',
    ]

    xml_lines = [
        f'<mujoco model="{model_name}">',
        "  <asset>",
        *asset_lines,
        "  </asset>",
        "",
        "  <worldbody>",
        "    <body>",
        '      <body name="object">',
        *collision_geom_lines,
        "",
        *visual_geom_lines,
        "      </body>",
        "",
        f'      <site name="bottom_site" rgba="0 0 0 0" size="0.005" pos="0 0 {bottom_z:g}"/>',
        f'      <site name="top_site" rgba="0 0 0 0" size="0.005" pos="0 0 {top_z:g}"/>',
        f'      <site name="horizontal_radius_site" rgba="0 0 0 0" size="0.005" pos="{horizontal_radius:g} 0 0"/>',
        "    </body>",
        "  </worldbody>",
        "</mujoco>",
        "",
    ]
    return "\n".join(xml_lines)


def fmt_vec(values) -> str:
    return " ".join(f"{float(v):g}" for v in values)


def build_hecrl_outer_xml(
    asset_name: str,
    visual_mesh_name: str,
    collision_mesh_names: list[str],
    texture_name: str,
    material_name: str,
    texture_file_name: str,
    visual_file_name: str,
    collision_file_names: list[str],
    scale_str: str,
    density: float,
    friction: str,
    reflectance: float,
    texrepeat: str,
    geom_quat,
    geom_z_offset: float,
    collision_group: int,
) -> str:
    collision_class = f"{asset_name}_collision"
    visual_class = f"{asset_name}_visual"
    target_class = f"{asset_name}_target"
    site_class = f"{asset_name}_com"
    marker_site_class = f"{asset_name}_marker_site"
    geom_quat_str = fmt_vec(normalize_wxyz_quat(geom_quat))
    geom_pos_str = f"0 0 {geom_z_offset:g}"

    collision_asset_lines = [
        f'    <mesh name="{mesh_name}" file="{file_name}" scale="{scale_str}"/>'
        for mesh_name, file_name in zip(collision_mesh_names, collision_file_names)
    ]

    return dedent(
        f"""\
        <mujoco model="{asset_name}_outer">
          <asset>
            <mesh name="{visual_mesh_name}" file="{visual_file_name}" scale="{scale_str}"/>
{chr(10).join(collision_asset_lines)}
            <texture name="{texture_name}" type="2d" file="{texture_file_name}"/>
            <material name="{material_name}"
                      texture="{texture_name}"
                      rgba="1 1 1 1"
                      texrepeat="{texrepeat}"
                      texuniform="false"
                      reflectance="{reflectance:g}"/>
          </asset>

          <default>
            <default class="{collision_class}">
              <geom type="mesh"
                    pos="{geom_pos_str}"
                    quat="{geom_quat_str}"
                    density="{density:g}"
                    friction="{friction}"
                    solimp="0.998 0.998 0.001"
                    solref="0.001 1"
                    rgba="0 0 0 0"
                    contype="1"
                    conaffinity="1"
                    group="{int(collision_group)}"
                    condim="4"/>
            </default>
            <default class="{visual_class}">
              <geom type="mesh"
                    mesh="{visual_mesh_name}"
                    pos="{geom_pos_str}"
                    quat="{geom_quat_str}"
                    material="{material_name}"
                    rgba="1 1 1 1"
                    contype="0"
                    conaffinity="0"
                    group="1"/>
            </default>
            <default class="{target_class}">
              <geom type="mesh"
                    mesh="{visual_mesh_name}"
                    pos="{geom_pos_str}"
                    quat="{geom_quat_str}"
                    material="{material_name}"
                    rgba="0.5 0.5 0.5 0.2"
                    contype="0"
                    conaffinity="0"
                    group="1"/>
            </default>
            <default class="{site_class}">
              <site type="sphere" size="0.005" rgba="0 1 0 1" group="5"/>
            </default>
            <default class="{marker_site_class}">
              <site type="sphere" size="0.005" rgba="0 0 0 0" group="5"/>
            </default>
          </default>

          <worldbody>
          </worldbody>
        </mujoco>
        """
    )


def build_hecrl_inner_xml(
    asset_name: str,
    collision_mesh_names: list[str],
    bottom_z: float,
    top_z: float,
    horizontal_radius: float,
    object_z: float,
    target_z: float,
) -> str:
    collision_class = f"{asset_name}_collision"
    visual_class = f"{asset_name}_visual"
    target_class = f"{asset_name}_target"
    site_class = f"{asset_name}_com"
    marker_site_class = f"{asset_name}_marker_site"
    collision_lines = [
        f'              <geom name="object_0_collision_{i}" class="{collision_class}" mesh="{mesh_name}"/>'
        for i, mesh_name in enumerate(collision_mesh_names)
    ]

    return dedent(
        f"""\
        <mujoco model="{asset_name}_inner">
          <worldbody>
            <body name="object_0" pos="0.3 0 {object_z:g}">
              <freejoint name="object_joint_0"/>
{chr(10).join(collision_lines)}
              <geom name="object_0_visual" class="{visual_class}"/>
              <site name="com_0" class="{site_class}"/>
              <site name="bottom_site_0" class="{marker_site_class}" pos="0 0 {bottom_z:g}"/>
              <site name="top_site_0" class="{marker_site_class}" pos="0 0 {top_z:g}"/>
              <site name="horizontal_radius_site_0" class="{marker_site_class}" pos="{horizontal_radius:g} 0 0"/>
            </body>
            <body name="object_target_0" pos="0.45 0 {target_z:g}" mocap="true">
              <geom name="target_object_0" class="{target_class}"/>
            </body>
          </worldbody>
        </mujoco>
        """
    )


def main():
    ap = argparse.ArgumentParser(
        description="Generate a robosuite-compatible GLB object asset using CoACD for multi-convex collision."
    )
    ap.add_argument("--dataset-root", default="glb_assets")
    ap.add_argument("--asset-name", help="Input folder name under glb_assets/. Defaults to --name.")
    ap.add_argument("--glb-path", default=None, help="Exact GLB file to use. Overrides automatic search under --asset-name.")
    ap.add_argument("--name", required=True, help="Object name in the form <category><index>, for example bowl01.")
    ap.add_argument("--extra-assets-root", default="extra_assets/glb_style")
    ap.add_argument(
        "--hecrl-assets-root",
        default="local_object_packs/assets",
        help="Output root for HECRL dual XML assets. Writes <root>/<category>/outer.xml and inner.xml.",
    )
    ap.add_argument(
        "--skip-hecrl-export",
        action="store_true",
        help="Only write the robosuite-style object.xml export, not HECRL outer.xml/inner.xml assets.",
    )
    ap.add_argument("--defs", default="extra_assets/scenes/g1/non_robot_defs.xml")
    ap.add_argument("--layout", default="extra_assets/scenes/g1/non_robot_layout.xml")
    ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--size", choices=list(SIZE_PRESETS.keys()), default="very_large")
    ap.add_argument("--mask-output", default="001.png")
    ap.add_argument("--mask-size", type=int, default=768)
    ap.add_argument("--density", type=float, default=300.0)
    ap.add_argument("--friction", default="1 0.3 0.1")
    ap.add_argument("--reflectance", type=float, default=0.5)
    ap.add_argument("--texrepeat", default="1 1")
    ap.add_argument(
        "--pos",
        type=float,
        nargs=3,
        default=[-2.8, 0.0, 0.5],
        help="Legacy scene-layout position. The current robosuite object.xml export does not use this value.",
    )
    ap.add_argument(
        "--quat",
        type=float,
        nargs=4,
        default=None,
        help="Local WXYZ quaternion written onto all generated GLB mesh geoms and used for the preview mask. Defaults can be asset-specific.",
    )
    ap.add_argument(
        "--geom-z-offset",
        type=float,
        default=None,
        help="Local z-offset written onto all generated GLB mesh geoms. By default this is computed so the mesh bottom matches bottom-z.",
    )
    ap.add_argument("--bottom-z", type=float, default=None)
    ap.add_argument("--top-z", type=float, default=None)
    ap.add_argument("--horizontal-radius", type=float, default=None)
    ap.add_argument("--threshold", type=float, default=0.05)
    ap.add_argument("--max-convex-hull", type=int, default=-1)
    ap.add_argument("--preprocess-mode", default="auto", choices=["auto", "on", "off"])
    ap.add_argument("--preprocess-resolution", type=int, default=50)
    ap.add_argument("--resolution", type=int, default=2000)
    ap.add_argument("--mcts-nodes", type=int, default=20)
    ap.add_argument("--mcts-iterations", type=int, default=150)
    ap.add_argument("--mcts-max-depth", type=int, default=3)
    ap.add_argument("--pca", action="store_true")
    ap.add_argument("--no-merge", action="store_true")
    ap.add_argument("--decimate", action="store_true")
    ap.add_argument("--max-ch-vertex", type=int, default=256)
    ap.add_argument("--extrude", action="store_true")
    ap.add_argument("--extrude-margin", type=float, default=0.01)
    ap.add_argument("--apx-mode", default="ch", choices=["ch", "box"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--real-metric", action="store_true")
    ap.add_argument("--no-flip-v", action="store_true", help="Do not flip GLB texture V coordinates when writing OBJ.")
    ap.add_argument(
        "--visual-max-faces",
        type=int,
        default=5000,
        help="Decimate runtime visual OBJ to at most this many faces with Blender. Use <=0 to keep full mesh.",
    )
    ap.add_argument(
        "--texture-max-size",
        type=int,
        default=1024,
        help="Resize the longest side of the runtime texture to this many pixels. Use <=0 to keep the original texture.",
    )
    ap.add_argument("--blender-bin", default=None, help="Optional path to Blender for visual mesh decimation.")
    ap.add_argument("--collision-group", type=int, default=3, help="MuJoCo geom group for invisible HECRL collision geoms.")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    dataset_root = Path(args.dataset_root).resolve()
    raw_name = args.name
    asset_name, variant_id = split_numbered_name(raw_name)
    input_asset_name = safe_name(args.asset_name or raw_name)
    asset_root = (dataset_root / input_asset_name).resolve()
    if not asset_root.exists():
        raise FileNotFoundError(f"Asset folder not found: {asset_root}")

    if args.glb_path is None:
        glb_path = find_glb_for_asset(asset_root)
    else:
        glb_path = Path(args.glb_path).expanduser()
        if not glb_path.is_absolute():
            glb_path = (Path.cwd() / glb_path).resolve()
        if not glb_path.exists():
            raise FileNotFoundError(f"GLB file not found: {glb_path}")
    object_name = safe_name(raw_name)
    model_name = f"{asset_name}_{variant_id}"
    size_scale = SIZE_PRESETS[args.size]
    effective_scale = args.scale * size_scale

    gltf, bin_chunk = parse_glb(glb_path)
    primitives = collect_primitives(gltf, bin_chunk)
    if not primitives:
        raise ValueError(f"No mesh primitives found in {glb_path}")

    object_quat = normalize_wxyz_quat(args.quat if args.quat is not None else default_glb_object_quat_for_asset(asset_name))
    mins, maxs, computed_horizontal_radius = compute_bounds(primitives, quat=object_quat)
    scaled_mins = mins * effective_scale
    scaled_maxs = maxs * effective_scale
    ycb_site_profile = get_ycb_site_profile(asset_name)

    bottom_z = args.bottom_z if args.bottom_z is not None else (
        ycb_site_profile["bottom_z"]
        if ycb_site_profile is not None
        else float(-0.5 * (scaled_maxs[2] - scaled_mins[2]))
    )
    geom_z_offset = args.geom_z_offset if args.geom_z_offset is not None else float(bottom_z - scaled_mins[2])
    if args.geom_z_offset is not None and args.bottom_z is None:
        bottom_z = float(scaled_mins[2] + geom_z_offset)

    top_z = args.top_z if args.top_z is not None else (
        ycb_site_profile["top_z"]
        if ycb_site_profile is not None and "top_z" in ycb_site_profile
        else float(scaled_maxs[2] + geom_z_offset)
    )
    horizontal_radius = args.horizontal_radius if args.horizontal_radius is not None else (
        ycb_site_profile["horizontal_radius"]
        if ycb_site_profile is not None and "horizontal_radius" in ycb_site_profile
        else float(computed_horizontal_radius * effective_scale)
    )
    object_z = float(max(0.0, -bottom_z))
    target_z = object_z

    texture_bytes, texture_ext = extract_base_color_texture(gltf, bin_chunk)
    texture_bytes = resize_texture_bytes(texture_bytes, texture_ext, args.texture_max_size)
    vertices, faces = merge_primitives(primitives)
    hulls = run_coacd_decomposition(vertices, faces, args)
    if not hulls:
        raise RuntimeError("CoACD returned no convex hulls")

    extra_assets_root = Path(args.extra_assets_root).resolve()
    output_asset_dir = extra_assets_root / asset_name / variant_id
    texture_file_name = f"{asset_name}_texture_map{texture_ext}"
    visual_file_name = f"{asset_name}_textured.obj"
    xml_file_name = "object.xml"

    visual_obj_path = output_asset_dir / visual_file_name
    texture_path = output_asset_dir / texture_file_name
    xml_path = output_asset_dir / xml_file_name
    mask_output_path = output_asset_dir / args.mask_output

    blender_bin = args.blender_bin or shutil.which("blender")
    visual_export_summary = write_visual_obj_for_runtime(
        visual_obj_path,
        primitives,
        overwrite=args.overwrite,
        flip_v=not args.no_flip_v,
        max_faces=args.visual_max_faces,
        blender_bin=blender_bin,
    )
    write_bytes(texture_path, texture_bytes, overwrite=args.overwrite)
    render_coacd_mask(hulls, mask_output_path, effective_scale, args.mask_size, object_quat)

    collision_mesh_names = []
    collision_file_names = []
    for i, (hull_vertices, hull_faces) in enumerate(hulls):
        collision_file_name = f"{asset_name}_collision_{i}.obj"
        collision_obj_path = output_asset_dir / collision_file_name
        write_collision_obj(collision_obj_path, np.asarray(hull_vertices), np.asarray(hull_faces), overwrite=args.overwrite)
        mesh_name = f"{object_name}_collision_mesh_{i}"
        collision_mesh_names.append(mesh_name)
        collision_file_names.append(collision_file_name)

    scale_str = f"{effective_scale:g} {effective_scale:g} {effective_scale:g}"
    visual_mesh_name = f"{object_name}_visual_mesh"
    texture_name = f"tex-{object_name}"
    material_name = f"{object_name}_material"
    xml_content = build_object_xml(
        model_name=model_name,
        visual_mesh_name=visual_mesh_name,
        collision_mesh_names=collision_mesh_names,
        texture_name=texture_name,
        material_name=material_name,
        texture_file_name=texture_path.name,
        visual_file_name=visual_obj_path.name,
        collision_file_names=collision_file_names,
        scale_str=scale_str,
        density=args.density,
        friction=args.friction,
        reflectance=args.reflectance,
        texrepeat=args.texrepeat,
        bottom_z=bottom_z,
        top_z=top_z,
        horizontal_radius=horizontal_radius,
        geom_quat=object_quat,
        geom_z_offset=geom_z_offset,
        collision_group=args.collision_group,
    )
    write_text(xml_path, xml_content, overwrite=args.overwrite)

    hecrl_asset_dir = None
    hecrl_inner_path = None
    hecrl_outer_path = None
    if not args.skip_hecrl_export:
        hecrl_assets_root = Path(args.hecrl_assets_root).resolve()
        hecrl_asset_dir = hecrl_assets_root / asset_name
        hecrl_visual_obj_path = hecrl_asset_dir / visual_file_name
        hecrl_texture_path = hecrl_asset_dir / texture_file_name
        hecrl_inner_path = hecrl_asset_dir / "inner.xml"
        hecrl_outer_path = hecrl_asset_dir / "outer.xml"

        write_bytes(hecrl_visual_obj_path, visual_obj_path.read_bytes(), overwrite=args.overwrite)
        write_bytes(hecrl_texture_path, texture_bytes, overwrite=args.overwrite)

        for i, (hull_vertices, hull_faces) in enumerate(hulls):
            collision_obj_path = hecrl_asset_dir / collision_file_names[i]
            write_collision_obj(
                collision_obj_path,
                np.asarray(hull_vertices),
                np.asarray(hull_faces),
                overwrite=args.overwrite,
            )

        hecrl_outer_content = build_hecrl_outer_xml(
            asset_name=asset_name,
            visual_mesh_name=visual_mesh_name,
            collision_mesh_names=collision_mesh_names,
            texture_name=texture_name,
            material_name=material_name,
            texture_file_name=hecrl_texture_path.name,
            visual_file_name=hecrl_visual_obj_path.name,
            collision_file_names=collision_file_names,
            scale_str=scale_str,
            density=args.density,
            friction=args.friction,
            reflectance=args.reflectance,
            texrepeat=args.texrepeat,
            geom_quat=object_quat,
            geom_z_offset=geom_z_offset,
            collision_group=args.collision_group,
        )
        hecrl_inner_content = build_hecrl_inner_xml(
            asset_name=asset_name,
            collision_mesh_names=collision_mesh_names,
            bottom_z=bottom_z,
            top_z=top_z,
            horizontal_radius=horizontal_radius,
            object_z=object_z,
            target_z=target_z,
        )
        write_text(hecrl_outer_path, hecrl_outer_content, overwrite=args.overwrite)
        write_text(hecrl_inner_path, hecrl_inner_content, overwrite=args.overwrite)

    print("Generated GLB-derived object with CoACD:")
    print(f"  Asset      : {asset_root}")
    print(f"  GLB        : {glb_path}")
    print(f"  Robosuite Dir : {output_asset_dir}")
    print(f"  Robosuite XML : {xml_path}")
    if hecrl_asset_dir is not None:
        print(f"  HECRL Dir     : {hecrl_asset_dir}")
        print(f"  HECRL OUTER   : {hecrl_outer_path}")
        print(f"  HECRL INNER   : {hecrl_inner_path}")
    print(f"  Visual     : {visual_obj_path}")
    print(f"  Visual Mesh: {visual_export_summary}")
    print(f"  Texture    : {texture_path}")
    print(f"  Mask PNG   : {mask_output_path}")
    print(f"  Hulls      : {len(hulls)}")
    print(f"  Size       : {args.size} (scale={effective_scale:g})")
    print(f"  Bounds     : min={mins.tolist()} max={maxs.tolist()}")
    if ycb_site_profile is not None:
        print(f"  Bottom Reference : category-default ({asset_name})")
    else:
        print("  Bottom Reference : computed-from-rotated-bounds")
    print(f"  Geom Offset: z={geom_z_offset:g}")
    print(f"  Sites      : bottom={bottom_z:g} top={top_z:g} radius={horizontal_radius:g}")
    print(f"  Body z     : object={object_z:g} target={target_z:g}")
    print(f"  Style Slot : {asset_name}/{variant_id}")


if __name__ == "__main__":
    main()
