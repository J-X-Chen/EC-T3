import argparse
import base64
import json
import mimetypes
import os
import re
import time
import urllib.parse
import urllib.request
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any

import requests

from prompts.single_obj_prompt import build_single_obj_prompt


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_TEST_DIR = BASE_DIR / "glb_asset"
DEFAULT_MODEL = "doubao-seed3d-2-0-260328"
DEFAULT_OPTIONS = "--subdivisionlevel medium --fileformat glb"
DEFAULT_GENERATOR_MODEL = "Qwen/Qwen-Image"
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
SUCCESS_STATUSES = {"succeeded", "success"}
DONE_STATUSES = SUCCESS_STATUSES | {"failed", "cancelled", "canceled"}
CORE_STYLE_POOL = [
"plain white ceramic",
"blue-and-white porcelain",
"branded commercial product style",
"black matte ceramic",
"floral ceramic pattern",
"cute animal illustration style",
"pastel colorful ceramic",
"retro diner style",
"natural wood tableware",
"frosted glass",
]
# CUTLERY_STYLE_POOL = [
#     "stainless steel",
#     "brushed stainless steel",
#     "matte black metal",
#     "mirror polished metal",
#     "gold-tone cutlery",
#     "rose gold cutlery",
#     "minimal modern cutlery",
#     "Nordic minimalist cutlery",
#     "Japanese-style metal cutlery",
#     "restaurant-style cutlery",
#     "fine dining cutlery",
#     "vintage-style silverware",
#     "industrial kitchen style",
#     "camping utensil style",
#     "wooden handle cutlery",
#     "cute engraved cutlery",
#     "pastel handle cutlery",
#     "colorful modern cutlery",
#     "retro diner cutlery",
#     "matte textured metal",
# ]

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate 3D models from local images in glb_asset."
    )
    parser.add_argument(
        "--test-dir",
        default=str(DEFAULT_TEST_DIR),
        help="Directory for input images and output 3D files.",
    )
    parser.add_argument(
        "--image",
        help="Input image name/path.",
    )
    parser.add_argument(
        "--is_text",
        "--is-text",
        dest="is_text",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--text",
        default="",
        help="Object text used to generate an image when --test-dir has no input images.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Generate one task for every image found in --test-dir.",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--options", default=DEFAULT_OPTIONS)
    parser.add_argument("--poll-interval", type=float, default=10.0)
    parser.add_argument("--max-wait", type=float, default=1200.0)
    parser.add_argument(
        "--no-extract",
        action="store_true",
        help="Keep downloaded zip files without extracting them.",
    )
    parser.add_argument(
        "--generator_name",
        default=DEFAULT_GENERATOR_MODEL,
        help="Text-to-image model name used when --text triggers image generation.",
    )
    parser.add_argument("--prompt", default="", help="Override the single-object text-to-image prompt.")
    parser.add_argument("--prompt_file", default="", help="Load the text-to-image prompt from a txt file.")
    parser.add_argument(
        "--style",
        default="",
        help=(
            "Optional style/material for the text-to-image prompt. "
            f"Use 0-{len(CORE_STYLE_POOL) - 1} to choose CORE_STYLE_POOL, "
            "or pass custom text directly."
        ),
    )
    parser.add_argument("--negative_prompt", default="", help="Negative prompt for text-to-image generation.")
    parser.add_argument("--seed", type=int, default=None, help="Optional seed for text-to-image generation.")
    parser.add_argument(
        "--image_size",
        default="",
        help='Text-to-image size, e.g. "1328x1328". Some models may not support this field.',
    )
    parser.add_argument("--num_inference_steps", type=int, default=None)
    parser.add_argument("--guidance_scale", type=float, default=None)
    parser.add_argument("--cfg", type=float, default=None)
    parser.add_argument("--timeout", type=int, default=180, help="Text-to-image request timeout seconds.")
    parser.add_argument("--debug", action="store_true", help="Print text-to-image payload/response on failure.")
    return parser.parse_args()


def now_ts() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def safe_json(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, indent=2)


def resolve_style(style_arg: str) -> str:
    style_arg = (style_arg or "").strip()
    if not style_arg:
        return ""

    if re.fullmatch(r"\d+", style_arg):
        style_number = int(style_arg)
        if 0 <= style_number < len(CORE_STYLE_POOL):
            return CORE_STYLE_POOL[style_number]
        raise ValueError(
            f"--style index out of range: {style_number}. "
            f"Use 0-{len(CORE_STYLE_POOL) - 1}, "
            "or pass custom style text."
        )

    return style_arg


def apply_style_to_prompt(prompt: str, style: str) -> str:
    prompt = prompt.strip()
    style = style.strip()
    if not style:
        return prompt

    return (
        f"{prompt}\n\n"
        f"Required object style/material: {style}.\n\n"
        "The object must clearly follow this style, material, color palette, "
        "surface finish, and design language while remaining realistic, "
        "functional, and physically plausible."
    )


def load_text_prompt(args: argparse.Namespace, object_text: str, style: str = "") -> str:
    if args.prompt and args.prompt.strip():
        return apply_style_to_prompt(args.prompt, style)

    if args.prompt_file:
        prompt_path = Path(args.prompt_file)
        txt = prompt_path.read_text(encoding="utf-8").strip()
        if txt:
            return apply_style_to_prompt(txt, style)

    return apply_style_to_prompt(build_single_obj_prompt(object_text), style)


def configure_text_generator(args: argparse.Namespace) -> argparse.Namespace:
    if args.generator_name == "Qwen/Qwen-Image":
        args.img_api_url = "https://api.siliconflow.cn/v1/images/generations"
        args.generator_api_key = os.environ.get("SILICONFLOW_API_KEY")
    elif args.generator_name == "gpt-4-32k":
        args.img_api_url = "https://aigc-api.hkust-gz.edu.cn/v1/chat/completions"
        args.generator_api_key = os.environ.get("OPENAI_API_KEY")
    elif args.generator_name == "grok-imagine-0.9":
        args.img_api_url = "https://ai.xiaoxinapi.com/v1/images/generations"
        args.generator_api_key = os.environ.get("XIAOXIN_API_KEY")
    elif args.generator_name == "dall-e-3":
        args.img_api_url = "https://ai.xiaoxinapi.com/v1/images/generations"
        args.generator_api_key = os.environ.get("XIAOXIN_API_KEY")
    elif args.generator_name == "Kwai-Kolors/Kolors":
        args.img_api_url = "https://api.siliconflow.cn/v1/images/generations"
        args.generator_api_key = os.environ.get("SILICONFLOW_API_KEY")
    elif args.generator_name == "qwen-image-max":
        args.img_api_url = "https://dashscope.aliyuncs.com/api/v1/services/aigc/multimodal-generation/generation"
        args.generator_api_key = os.getenv("DASHSCOPE_API_KEY")
    elif args.generator_name in {"gpt-image-2-2", "gpt-image-2"}:
        args.img_api_url = "https://az.gptplus5.com/v1/images/generations"
        args.generator_api_key = os.getenv("XUNXAI_KEY")
    else:
        raise ValueError(f"Unknown generator model: {args.generator_name}")

    return args


def is_dashscope_qwen_image_request(api_url: str, model: str) -> bool:
    return (
        api_url == "https://dashscope.aliyuncs.com/api/v1/services/aigc/multimodal-generation/generation"
        and model.startswith("qwen-image")
    )


def build_t2i_payload(
    api_url: str,
    model: str,
    prompt: str,
    negative_prompt: str = "",
    seed: int | None = None,
    image_size: str = "",
    num_inference_steps: int | None = None,
    guidance_scale: float | None = None,
    cfg: float | None = None,
) -> dict[str, Any]:
    if is_dashscope_qwen_image_request(api_url, model):
        payload: dict[str, Any] = {
            "model": model,
            "input": {
                "messages": [
                    {
                        "role": "user",
                        "content": [{"text": prompt}],
                    }
                ]
            },
        }
        parameters: dict[str, Any] = {}
        if negative_prompt:
            parameters["negative_prompt"] = negative_prompt
        if seed is not None:
            parameters["seed"] = int(seed)
        if image_size:
            parameters["size"] = image_size.replace("x", "*").replace("X", "*")
        if num_inference_steps is not None:
            parameters["steps"] = int(num_inference_steps)
        if parameters:
            payload["parameters"] = parameters
        return payload

    payload = {
        "model": model,
        "prompt": prompt,
    }
    if negative_prompt:
        payload["negative_prompt"] = negative_prompt
    if seed is not None:
        payload["seed"] = int(seed)
    if image_size:
        payload["image_size"] = image_size
    if num_inference_steps is not None:
        payload["num_inference_steps"] = int(num_inference_steps)
    if guidance_scale is not None:
        payload["guidance_scale"] = float(guidance_scale)
    if cfg is not None:
        payload["cfg"] = float(cfg)
    return payload


def extract_t2i_images(api_url: str, model: str, meta: dict[str, Any]) -> list[dict[str, str]]:
    images = meta.get("image", [])
    if images:
        return images

    data = meta.get("data", [])
    if isinstance(data, list):
        parsed_images: list[dict[str, str]] = []
        for item in data:
            if not isinstance(item, dict):
                continue

            if item.get("url"):
                parsed_images.append({"url": item["url"]})
                continue

            for key in ("b64_json", "b64", "image_base64", "base64"):
                if item.get(key):
                    parsed_images.append({"b64_json": item[key]})
                    break

        if parsed_images:
            return parsed_images

    if is_dashscope_qwen_image_request(api_url, model):
        output = meta.get("output", {})
        if isinstance(output, dict):
            choices = output.get("choices", [])
            if isinstance(choices, list):
                parsed_images: list[dict[str, str]] = []
                for choice in choices:
                    if not isinstance(choice, dict):
                        continue
                    message = choice.get("message", {})
                    if not isinstance(message, dict):
                        continue
                    content = message.get("content", [])
                    if not isinstance(content, list):
                        continue
                    for item in content:
                        if isinstance(item, dict) and item.get("image"):
                            parsed_images.append({"url": item["image"]})
                if parsed_images:
                    return parsed_images

    return []


def decode_base64_image(image_data: str) -> bytes:
    if not isinstance(image_data, str) or not image_data.strip():
        raise ValueError("image_data must be a non-empty base64 string")

    image_data = image_data.strip()
    if image_data.startswith("data:"):
        _, _, image_data = image_data.partition(",")

    return base64.b64decode(image_data)


def download_t2i_image(url: str, timeout: int) -> bytes:
    if url.startswith("data:"):
        return decode_base64_image(url)

    response = requests.get(url, timeout=timeout)
    response.raise_for_status()
    return response.content


def t2i_call_once(
    api_url: str,
    api_key: str,
    model: str,
    prompt: str,
    negative_prompt: str = "",
    seed: int | None = None,
    image_size: str = "",
    num_inference_steps: int | None = None,
    guidance_scale: float | None = None,
    cfg: float | None = None,
    timeout: int = 180,
    debug: bool = False,
) -> tuple[list[tuple[bytes, str]], dict[str, Any], dict[str, Any]]:
    if not api_key:
        raise RuntimeError("Missing text-to-image API key")
    if not prompt or not prompt.strip():
        raise ValueError("Prompt is empty. Provide --prompt, --prompt_file, or --text.")

    payload = build_t2i_payload(
        api_url=api_url,
        model=model,
        prompt=prompt,
        negative_prompt=negative_prompt,
        seed=seed,
        image_size=image_size,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        cfg=cfg,
    )

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    response = requests.post(api_url, headers=headers, json=payload, timeout=timeout)
    if response.status_code != 200:
        if debug:
            print("\n===== DEBUG: text-to-image model =====")
            print(model)
            print("\n===== DEBUG: payload sent =====")
            print(safe_json(payload))
            print("\n===== DEBUG: status / response =====")
            print("status:", response.status_code)
            print(response.text)
        response.raise_for_status()

    meta = response.json()
    images = extract_t2i_images(api_url=api_url, model=model, meta=meta)
    if not images:
        raise RuntimeError("No images returned:\n" + safe_json(meta))

    downloaded: list[tuple[bytes, str]] = []
    for item in images:
        url = item.get("url")
        if url:
            downloaded.append((download_t2i_image(url, timeout), url))
            continue

        b64_image = item.get("b64_json")
        if b64_image:
            downloaded.append((decode_base64_image(b64_image), "base64://inline"))

    if not downloaded:
        raise RuntimeError("Images returned but no valid image payload found:\n" + safe_json(meta))

    return downloaded, meta, payload


def image_suffix_from_bytes(image_bytes: bytes) -> str:
    if image_bytes.startswith(b"\x89PNG\r\n\x1a\n"):
        return ".png"
    if image_bytes.startswith(b"\xff\xd8\xff"):
        return ".jpg"
    if image_bytes.startswith(b"RIFF") and image_bytes[8:12] == b"WEBP":
        return ".webp"
    if image_bytes.startswith(b"BM"):
        return ".bmp"
    return ".png"


def text_input_from_args(args: argparse.Namespace) -> str:
    object_text = args.text.strip()
    if object_text:
        return object_text
    raise ValueError("Text-to-image generation requires --text when no input image is found.")


def generate_image_from_text(args: argparse.Namespace, output_dir: Path) -> Path:
    args = configure_text_generator(args)
    if not args.generator_api_key:
        raise RuntimeError(f"Missing API key for text-to-image generator: {args.generator_name}")

    object_text = text_input_from_args(args)
    style = resolve_style(args.style)
    prompt = load_text_prompt(args, object_text, style)
    print(f"Generating image from text: {object_text}")
    print(f"Text-to-image generator: {args.generator_name}")
    if style:
        print(f"Prompt style: {style}")

    downloaded, meta, used_payload = t2i_call_once(
        api_url=args.img_api_url,
        api_key=args.generator_api_key,
        model=args.generator_name,
        prompt=prompt,
        negative_prompt=args.negative_prompt,
        seed=args.seed,
        image_size=args.image_size,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        cfg=args.cfg,
        timeout=args.timeout,
        debug=args.debug,
    )

    image_bytes, _image_url = downloaded[0]
    output_dir.mkdir(parents=True, exist_ok=True)
    suffix = image_suffix_from_bytes(image_bytes)
    image_path = output_dir / f"{safe_name(object_text)}_{now_ts()}{suffix}"
    image_path.write_bytes(image_bytes)

    metadata_path = image_path.with_suffix(".t2i.json")
    metadata_path.write_text(
        safe_json(
            {
                "object_text": object_text,
                "style": style,
                "prompt": prompt,
                "generator_name": args.generator_name,
                "api_url": args.img_api_url,
                "payload": used_payload,
                "response": meta,
            }
        ),
        encoding="utf-8",
    )

    print(f"Generated image saved: {image_path}")
    print(f"Text-to-image metadata saved: {metadata_path}")
    return image_path


def fix_broken_ssl_cert_env() -> None:
    certifi_path: Path | None = None

    for env_name in ("SSL_CERT_FILE", "REQUESTS_CA_BUNDLE"):
        if env_name not in os.environ:
            continue

        current_value = os.environ.get(env_name, "")
        if current_value and Path(current_value).exists():
            continue

        if certifi_path is None:
            try:
                import certifi
            except ImportError:
                certifi_path = Path()
            else:
                certifi_path = Path(certifi.where())

        if certifi_path and certifi_path.exists():
            os.environ[env_name] = str(certifi_path)
            print(f"{env_name} was invalid; using certifi CA bundle: {certifi_path}")
        else:
            os.environ.pop(env_name, None)
            print(f"{env_name} was invalid and has been removed.")


def image_to_data_url(image_path: Path) -> str:
    mime_type = mimetypes.guess_type(image_path.name)[0] or "image/png"
    image_b64 = base64.b64encode(image_path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{image_b64}"


def find_images(test_dir: Path, image_arg: str | None, use_all: bool) -> list[Path]:
    test_dir.mkdir(parents=True, exist_ok=True)

    if image_arg:
        image_path = Path(image_arg)
        candidates = [image_path] if image_path.is_absolute() else [test_dir / image_path, image_path]
        for candidate in candidates:
            if candidate.exists() and candidate.is_file():
                return [candidate.resolve()]
        raise FileNotFoundError(f"Input image not found: {image_arg}")

    images = sorted(
        path.resolve()
        for path in test_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    return images if use_all else images[:1]


def object_to_dict(value):
    if hasattr(value, "model_dump"):
        return value.model_dump()
    if hasattr(value, "dict"):
        return value.dict()
    if hasattr(value, "to_dict"):
        return value.to_dict()
    if isinstance(value, dict):
        return {key: object_to_dict(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [object_to_dict(item) for item in value]
    return value


def get_field(value, *field_names):
    for field_name in field_names:
        if isinstance(value, dict) and field_name in value:
            return value[field_name]
        if hasattr(value, field_name):
            return getattr(value, field_name)
    return None


def get_task_id(create_result) -> str:
    task_id = get_field(create_result, "id", "task_id")
    if not task_id:
        raise RuntimeError(f"Could not find task id in response: {create_result}")
    return str(task_id)


def normalize_remote_url(value) -> str | None:
    if isinstance(value, str) and value and not value.startswith("data:"):
        return value
    if isinstance(value, dict):
        return normalize_remote_url(value.get("url"))
    return None


def find_remote_url(value, preferred_keys: tuple[str, ...]) -> str | None:
    value = object_to_dict(value)

    if isinstance(value, dict):
        for key in preferred_keys:
            remote_url = normalize_remote_url(value.get(key))
            if remote_url:
                return remote_url
        for key, item in value.items():
            if key == "image_url":
                continue
            remote_url = find_remote_url(item, preferred_keys)
            if remote_url:
                return remote_url

    if isinstance(value, list):
        for item in value:
            remote_url = find_remote_url(item, preferred_keys)
            if remote_url:
                return remote_url

    return None


def get_file_url(task_result) -> str:
    content = get_field(task_result, "content")
    file_url = find_remote_url(content, ("file_url", "model_url", "output_url"))
    if not file_url:
        file_url = find_remote_url(content, ("url",))
    if not file_url:
        raise RuntimeError(f"Could not find output file URL in response: {task_result}")
    return file_url


def wait_for_task(client, task_id: str, poll_interval: float, max_wait: float):
    start_time = time.monotonic()
    while True:
        task_result = client.content_generation.tasks.get(task_id=task_id)
        status = str(get_field(task_result, "status") or "").lower()
        print(f"Task {task_id}: {status or 'unknown'}")

        if status in DONE_STATUSES:
            return task_result
        if time.monotonic() - start_time > max_wait:
            raise TimeoutError(f"Task {task_id} did not finish within {max_wait} seconds")
        time.sleep(poll_interval)


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._") or "seed3d"


def output_archive_path(output_dir: Path, image_path: Path, task_id: str, file_url: str) -> Path:
    parsed = urllib.parse.urlparse(file_url)
    suffix = Path(parsed.path).suffix or ".zip"
    name = f"{safe_name(image_path.stem)}_{safe_name(task_id)}{suffix}"
    return output_dir / name


def download_file(file_url: str, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(file_url, timeout=120) as response:
        output_path.write_bytes(response.read())


def extract_zip_if_needed(archive_path: Path) -> list[Path]:
    if not zipfile.is_zipfile(archive_path):
        return [archive_path]

    extract_dir = archive_path.with_suffix("")
    extract_dir.mkdir(parents=True, exist_ok=True)
    extract_root = extract_dir.resolve()

    with zipfile.ZipFile(archive_path) as zip_file:
        for member in zip_file.infolist():
            target = (extract_dir / member.filename).resolve()
            if extract_root != target and extract_root not in target.parents:
                raise RuntimeError(f"Unsafe zip entry: {member.filename}")
        zip_file.extractall(extract_dir)

    model_files = [
        path
        for path in extract_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in {".glb", ".gltf", ".obj", ".fbx", ".usdz"}
    ]
    return model_files or [extract_dir]


def save_task_metadata(output_dir: Path, image_path: Path, task_id: str, task_result) -> Path:
    metadata_path = output_dir / f"{safe_name(image_path.stem)}_{safe_name(task_id)}.json"
    metadata_path.write_text(
        json.dumps(object_to_dict(task_result), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return metadata_path


def create_seed3d_task(client, model: str, options: str, image_path: Path):
    print(f"Creating task from local image: {image_path}")
    return client.content_generation.tasks.create(
        model=model,
        content=[
            {"type": "text", "text": options},
            {
                "type": "image_url",
                "image_url": {"url": image_to_data_url(image_path)},
            },
        ],
    )


def create_ark_client(api_key: str):
    try:
        from volcenginesdkarkruntime import Ark
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency: pip install 'volcengine-python-sdk[ark]'"
        ) from exc

    return Ark(
        base_url="https://ark.cn-beijing.volces.com/api/v3",
        api_key=api_key,
    )


def run_for_image(client, args: argparse.Namespace, image_path: Path, output_dir: Path) -> None:
    create_result = create_seed3d_task(client, args.model, args.options, image_path)
    task_id = get_task_id(create_result)
    print(f"Created task: {task_id}")

    task_result = wait_for_task(client, task_id, args.poll_interval, args.max_wait)
    status = str(get_field(task_result, "status") or "").lower()
    metadata_path = save_task_metadata(output_dir, image_path, task_id, task_result)

    if status not in SUCCESS_STATUSES:
        error = get_field(task_result, "error")
        raise RuntimeError(f"Task {task_id} ended with status {status}: {error}")

    file_url = get_file_url(task_result)
    archive_path = output_archive_path(output_dir, image_path, task_id, file_url)
    download_file(file_url, archive_path)
    print(f"Downloaded output: {archive_path}")
    print(f"Saved metadata: {metadata_path}")

    if not args.no_extract:
        extracted_paths = extract_zip_if_needed(archive_path)
        for extracted_path in extracted_paths:
            print(f"Extracted model/output: {extracted_path}")


def main() -> None:
    args = parse_args()
    output_dir = Path(args.test_dir).resolve()
    fix_broken_ssl_cert_env()

    images = find_images(output_dir, args.image, args.all)
    if images:
        print(f"Found {len(images)} input image(s) in {output_dir}; skipping text-to-image generation.")
    elif args.text.strip():
        images = [generate_image_from_text(args, output_dir)]
    else:
        raise FileNotFoundError(
            f"No input image found in {output_dir}. "
            "Add image files to --test-dir or pass --text to generate one first."
        )

    api_key = os.getenv("ARK_API_KEY")
    if not api_key:
        raise SystemExit("Please set ARK_API_KEY before running this script.")

    client = create_ark_client(api_key)
    # import pdb; pdb.set_trace()
    for image_path in images:
        run_for_image(client, args, image_path, output_dir)


if __name__ == "__main__":
    main()
