"""Create a short 2D/2.5D dance animation from a Bathukamma photograph."""

from __future__ import annotations

import argparse
import bisect
import json
import math
import os
import random
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

try:
    import imageio.v2 as imageio
    import imageio_ffmpeg
    import numpy as np
    from PIL import Image, ImageDraw, ImageFilter, ImageFont, ImageStat
except ImportError as error:
    print(f"A required package is missing: {error}")
    print("Install the project dependencies with: python -m pip install -r requirements.txt")
    raise SystemExit(1) from error


ROOT_DIR = Path(__file__).resolve().parent
INPUT_DIR = ROOT_DIR / "input"
ASSETS_DIR = ROOT_DIR / "assets"
GENERATED_DIR = ROOT_DIR / "generated"
FRAMES_DIR = GENERATED_DIR / "frames"
OUTPUT_DIR = ROOT_DIR / "output"
IMAGE_PATH = INPUT_DIR / "women.png"
AUDIO_PATH = INPUT_DIR / "bathukamma_song.mp3"
ANALYSIS_PATH = GENERATED_DIR / "audio_analysis.json"
SILENT_VIDEO_PATH = GENERATED_DIR / "silent_video.mp4"
OUTPUT_VIDEO_PATH = OUTPUT_DIR / "bathukamma_dance.mp4"

WIDTH = 1280
HEIGHT = 720
FPS = 30
DURATION = 35.0
EXPECTED_DANCERS = 5
MAX_ROTATION = 6.0
_BACKGROUND_CACHE: dict[tuple[int, int], Image.Image] = {}

FLOWER_COLORS = [
    (242, 183, 54, 235),
    (209, 79, 74, 235),
    (236, 126, 49, 235),
    (223, 197, 95, 235),
    (167, 88, 129, 235),
    (237, 221, 184, 235),
]


class SegmentationError(RuntimeError):
    """Raised when neither automatic nor manual person layers are available."""


def ensure_directories() -> None:
    """Create every project folder without removing any existing files."""
    for directory in (INPUT_DIR, ASSETS_DIR, GENERATED_DIR, FRAMES_DIR, OUTPUT_DIR):
        directory.mkdir(parents=True, exist_ok=True)


def resolve_audio_path() -> Path:
    """Accept the requested filename and the alternate spelling of the supplied song."""
    candidates = (
        AUDIO_PATH,
        INPUT_DIR / "batukamma_song.mp3",
        ROOT_DIR / "bathukamma_song.mp3",
        ROOT_DIR / "batukamma_song.mp3",
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return AUDIO_PATH


def check_input_files() -> tuple[Path, Path]:
    """Check that the source photograph and song can be read."""
    audio_path = resolve_audio_path()
    if not IMAGE_PATH.is_file():
        raise FileNotFoundError(f"Please place your image at: {IMAGE_PATH.relative_to(ROOT_DIR)}")
    if not audio_path.is_file():
        raise FileNotFoundError("Please place your Bathukamma song at: input/bathukamma_song.mp3")
    try:
        with Image.open(IMAGE_PATH) as image:
            image.verify()
    except Exception as error:
        raise OSError("Unable to read input/women.png") from error
    if audio_path != AUDIO_PATH:
        print(f"Using supplied song file: {audio_path.relative_to(ROOT_DIR)}")
    return IMAGE_PATH, audio_path


def _feather_mask(mask: np.ndarray) -> Image.Image:
    """Soften only the cutout edge while keeping the original person pixels."""
    mask_image = Image.fromarray(mask.astype(np.uint8) * 255)
    try:
        import cv2

        softened = cv2.GaussianBlur(np.asarray(mask_image), (0, 0), 0.8)
        return Image.fromarray(softened)
    except ImportError:
        return mask_image.filter(ImageFilter.GaussianBlur(radius=0.8))


def _make_layer(
    source_image: Image.Image,
    mask: np.ndarray,
    bounding_box: tuple[float, float, float, float],
    source_index: int,
    source_width: int,
    source_height: int,
    hand_points: dict[str, tuple[float, float]] | None = None,
) -> dict[str, Any]:
    """Crop one detected dancer, retain original pixels, and add a feathered alpha mask."""
    left, top, right, bottom = bounding_box
    box_width = max(1.0, right - left)
    box_height = max(1.0, bottom - top)
    pad_x = max(8, int(box_width * 0.08))
    pad_y = max(8, int(box_height * 0.04))
    x0 = max(0, int(left) - pad_x)
    y0 = max(0, int(top) - pad_y)
    x1 = min(source_width, int(math.ceil(right)) + pad_x)
    y1 = min(source_height, int(math.ceil(bottom)) + pad_y)

    rgba = source_image.convert("RGBA").crop((x0, y0, x1, y1))
    alpha = _feather_mask(mask[y0:y1, x0:x1])
    rgba.putalpha(alpha)
    return {
        "image": rgba,
        "center_x": ((left + right) / 2) / source_width,
        "source_index": source_index,
        "crop_origin": (x0, y0),
        "hand_points": hand_points or {},
    }


def _detect_hand_points(
    image_path: Path,
    detections: list[dict[str, Any]],
) -> dict[int, dict[str, tuple[float, float]]]:
    """Match pose wrists to the five selected foreground segmentation boxes."""
    from ultralytics import YOLO

    prediction = YOLO("yolo11n-pose.pt").predict(
        source=str(image_path),
        classes=[0],
        conf=0.15,
        imgsz=960,
        verbose=False,
    )[0]
    if prediction.boxes is None or prediction.keypoints is None:
        return {}

    pose_boxes = prediction.boxes.xyxy.cpu().numpy()
    pose_points = prediction.keypoints.data.cpu().numpy()
    available = set(range(len(pose_boxes)))
    matched_hands: dict[int, dict[str, tuple[float, float]]] = {}

    for detection in detections:
        x1, y1, x2, y2 = detection["box"]
        detection_area = max(1.0, (x2 - x1) * (y2 - y1))
        candidates = []
        for pose_index in available:
            px1, py1, px2, py2 = pose_boxes[pose_index]
            overlap_width = max(0.0, min(x2, px2) - max(x1, px1))
            overlap_height = max(0.0, min(y2, py2) - max(y1, py1))
            intersection = overlap_width * overlap_height
            pose_area = max(1.0, (px2 - px1) * (py2 - py1))
            union = detection_area + pose_area - intersection
            candidates.append((intersection / max(1.0, union), pose_index))
        if not candidates:
            continue

        overlap, pose_index = max(candidates)
        if overlap < 0.10:
            continue
        available.remove(pose_index)
        wrists = []
        for wrist_index in (9, 10):
            wrist = pose_points[pose_index][wrist_index]
            if float(wrist[2]) >= 0.20:
                wrists.append((float(wrist[0]), float(wrist[1])))
        if wrists:
            matched_hands[detection["source_index"]] = {
                "screen_left": min(wrists, key=lambda point: point[0]),
                "screen_right": max(wrists, key=lambda point: point[0]),
            }

    return matched_hands


def _manual_person_layers() -> list[dict[str, Any]] | None:
    """Load five user-supplied transparent PNGs when automatic separation is unavailable."""
    paths = [ASSETS_DIR / f"person{index}.png" for index in range(1, EXPECTED_DANCERS + 1)]
    if not all(path.is_file() for path in paths):
        return None

    layers = []
    for index, path in enumerate(paths):
        with Image.open(path) as image:
            person = image.convert("RGBA")
        if person.getbbox() is None:
            print(f"Ignoring empty manual cutout: {path.name}")
            return None
        layers.append({
            "image": person,
            "center_x": (index + 0.5) / EXPECTED_DANCERS,
            "source_index": index,
        })
    print("Using the five manual transparent PNG cutouts in assets/.")
    return layers


def extract_people(image_path: Path) -> list[dict[str, Any]]:
    """Detect and save five person cutouts, or use the documented manual fallback."""
    with Image.open(image_path) as opened_image:
        source_image = opened_image.convert("RGB")
    source_width, source_height = source_image.size
    layers: list[dict[str, Any]] = []

    try:
        from ultralytics import YOLO

        print("Loading YOLO person-segmentation model (first run may download its weights)...")
        model = YOLO("yolov8n-seg.pt")
        predictions = model.predict(
            source=str(image_path),
            classes=[0],
            conf=0.20,
            retina_masks=True,
            verbose=False,
        )
        if predictions and predictions[0].boxes is not None and predictions[0].masks is not None:
            result = predictions[0]
            boxes = result.boxes.xyxy.cpu().numpy()
            confidences = result.boxes.conf.cpu().numpy()
            masks = result.masks.data.cpu().numpy()
            detections = []
            for index, (box, confidence, raw_mask) in enumerate(zip(boxes, confidences, masks)):
                mask = (raw_mask > 0.5).astype(np.uint8)
                if mask.shape != (source_height, source_width):
                    mask_image = Image.fromarray(mask * 255)
                    mask = np.asarray(
                        mask_image.resize((source_width, source_height), Image.Resampling.NEAREST)
                    ) > 0
                    mask = mask.astype(np.uint8)
                area = max(1, int(mask.sum()))
                detections.append({
                    "box": tuple(float(value) for value in box),
                    "confidence": float(confidence),
                    "mask": mask,
                    "score": area * float(confidence),
                    "source_index": index,
                })

            if len(detections) > EXPECTED_DANCERS:
                detections = sorted(detections, key=lambda item: item["score"], reverse=True)[
                    :EXPECTED_DANCERS
                ]
                print("More than five people were detected; selecting the five largest/highest-confidence detections.")
            if len(detections) == EXPECTED_DANCERS:
                detections.sort(key=lambda item: (item["box"][0] + item["box"][2]) / 2)
                try:
                    hand_points = _detect_hand_points(image_path, detections)
                    print(f"Matched hand landmarks for {len(hand_points)} foreground dancers.")
                except Exception as error:
                    hand_points = {}
                    print(f"Hand-pose detection unavailable; using the flower-circle layout. {error}")
                for item in detections:
                    layers.append(
                        _make_layer(
                            source_image,
                            item["mask"],
                            item["box"],
                            item["source_index"],
                            source_width,
                            source_height,
                            hand_points.get(item["source_index"]),
                        )
                    )
                for index, layer in enumerate(layers, start=1):
                    layer["image"].save(ASSETS_DIR / f"person{index}.png")
                print("Detected five people and saved transparent cutouts in assets/.")
                return layers
            print(f"Person segmentation found {len(detections)} people; five are required.")
    except Exception as error:
        message = str(error).replace("\n", " ")[:180]
        print(f"Automatic segmentation could not complete: {message}")

    manual_layers = _manual_person_layers()
    if manual_layers is not None:
        return manual_layers

    print("\n==================================================")
    print("AUTOMATIC PERSON SEGMENTATION FAILED")
    print("==================================================")
    print("Please provide five transparent PNG files:")
    for index in range(1, EXPECTED_DANCERS + 1):
        print(f"assets/person{index}.png")
    print("\nThen run:\npython main.py")
    raise SegmentationError("Five people could not be separated from the source image.")


def _probe_audio_duration(audio_path: Path) -> float:
    """Read the duration from FFmpeg metadata without decoding the full song."""
    try:
        ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()
        result = subprocess.run(
            [ffmpeg_path, "-hide_banner", "-i", str(audio_path)],
            capture_output=True,
            text=True,
            timeout=20,
            check=False,
        )
        match = re.search(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)", result.stderr)
        if match:
            hours, minutes, seconds = match.groups()
            return int(hours) * 3600 + int(minutes) * 60 + float(seconds)
    except Exception:
        return DURATION
    return DURATION


def analyze_audio(audio_path: Path) -> dict[str, Any]:
    """Analyze the song with librosa and write beat and duration details to JSON."""
    source_duration = _probe_audio_duration(audio_path)
    beat_times: list[float] = []
    bpm = 120.0
    used_fallback = False

    try:
        import librosa

        samples, sample_rate = librosa.load(str(audio_path), sr=22050, mono=True)
        if samples.size == 0:
            raise ValueError("The audio file contains no readable samples.")
        detected_tempo, beat_frames = librosa.beat.beat_track(
            y=samples,
            sr=sample_rate,
            units="frames",
        )
        tempo_values = np.asarray(detected_tempo).reshape(-1)
        if tempo_values.size and float(tempo_values[0]) > 0:
            bpm = float(tempo_values[0])
        beat_times = [
            float(value)
            for value in librosa.frames_to_time(beat_frames, sr=sample_rate)
            if float(value) < source_duration
        ]
        if not beat_times:
            raise ValueError("No beat positions were returned.")
    except Exception:
        print("Beat detection failed. Using fallback dance timing.")
        used_fallback = True
        beat_times = [round(index * 0.5, 3) for index in range(math.ceil(source_duration / 0.5))]

    analysis = {
        "bpm": round(bpm, 2),
        "duration": round(source_duration, 3),
        "beats": [round(value, 3) for value in beat_times],
        "beat_detection_fallback": used_fallback,
        "render_duration": DURATION,
    }
    ANALYSIS_PATH.write_text(json.dumps(analysis, indent=2), encoding="utf-8")
    print(f"Song duration: {source_duration:.1f} seconds; dance timing: {bpm:.0f} BPM.")
    return analysis


def smooth_step(t: float) -> float:
    """Return smooth interpolation for a value between zero and one."""
    return t * t * (3 - 2 * t)


def _beat_state(current_time: float, beat_times: list[float]) -> dict[str, float | int]:
    """Find the current beat, its phase, and a short accent around each beat."""
    if not beat_times:
        beat_times = [index * 0.5 for index in range(41)]
    beat_index = max(0, bisect.bisect_right(beat_times, current_time) - 1)
    current_beat = beat_times[min(beat_index, len(beat_times) - 1)]
    next_beat = beat_times[min(beat_index + 1, len(beat_times) - 1)]
    beat_length = max(0.25, next_beat - current_beat)
    phase = min(1.0, max(0.0, (current_time - current_beat) / beat_length))
    accent = max(0.0, 1.0 - (current_time - current_beat) / 0.16)
    return {"index": beat_index, "phase": phase, "accent": accent}


def group_dance(current_time: float, beat: dict[str, float | int]) -> dict[str, float]:
    """Calculate shared group movement for the current section of the timeline."""
    if current_time < 2.0:
        amount, speed = 3.0, 0.8
    elif current_time < 8.0:
        amount, speed = 9.0, 1.2
    elif current_time < 14.0:
        amount, speed = 10.0, 2.0
    elif current_time < 20.0:
        amount, speed = 6.0, 1.5
    elif current_time < 26.0:
        amount, speed = 13.0, 1.45
    elif current_time < 32.0:
        amount, speed = 15.0, 2.1
    elif current_time < 33.0:
        amount, speed = 3.0, 0.7
    else:
        amount, speed = 12.0, 1.8
    beat_accent = float(beat["accent"])
    beat_direction = -1.0 if int(beat["index"]) % 2 == 0 else 1.0
    shift = math.sin(current_time * speed) * amount + beat_direction * beat_accent * 4.0
    bounce = beat_accent * 5.0
    if 14.0 <= current_time < 20.0 or current_time >= 33.0:
        bounce += abs(math.sin(current_time * speed * 2.0)) * 8.0
    return {"shift": shift, "bounce": bounce, "speed": speed}


def sway_person(person_index: int, current_time: float, beat: dict[str, float | int]) -> float:
    """Move each dancer side to side, with the outer pair moving in opposition."""
    if current_time < 2.0:
        amplitude, speed = 6.0, 0.9
    elif current_time < 8.0:
        amplitude, speed = 11.0, 1.3
    elif current_time < 14.0:
        amplitude, speed = 18.0, 2.0
    elif current_time < 20.0:
        amplitude, speed = 9.0, 1.4
    elif current_time < 26.0:
        amplitude, speed = 14.0, 1.5
    elif current_time < 32.0:
        amplitude, speed = 19.0, 2.2
    elif current_time < 33.0:
        amplitude, speed = 4.0, 0.8
    else:
        amplitude, speed = 16.0, 1.9

    phase_offset = (person_index % 2) * math.pi
    if person_index in (1, 4):
        phase_offset = math.pi
    if person_index == 2:
        amplitude *= 0.45
    return math.sin(current_time * speed + phase_offset) * amplitude


def bounce_person(person_index: int, current_time: float, beat: dict[str, float | int]) -> float:
    """Give each full-person cutout a small, smooth vertical bounce."""
    speed = 2.0 if current_time < 26.0 else 2.7
    phase = current_time * speed + person_index * 0.32
    bounce = (1.0 - math.cos(phase)) * 2.0
    bounce += float(beat["accent"]) * (3.0 if person_index == 2 else 1.5)
    if 14.0 <= current_time < 20.0 or current_time >= 33.0:
        bounce += abs(math.sin(phase)) * 5.0
    return min(13.0, bounce)


def rotate_person(person_index: int, current_time: float, beat: dict[str, float | int]) -> float:
    """Return a gentle whole-person rotation that never exceeds six degrees."""
    phase = current_time * 1.7 + person_index * 0.6
    rotation = math.sin(phase) * 3.1
    rotation += (1 if int(beat["index"]) % 2 else -1) * float(beat["accent"]) * 0.8
    return max(-MAX_ROTATION, min(MAX_ROTATION, rotation))


def move_person(
    person_index: int,
    current_time: float,
    beat: dict[str, float | int],
    group: dict[str, float],
) -> tuple[float, float]:
    """Combine dancer-specific sway with safe, shared group movement."""
    x_movement = sway_person(person_index, current_time, beat) + group["shift"]
    y_movement = -bounce_person(person_index, current_time, beat) - group["bounce"]
    return max(-42.0, min(42.0, x_movement)), max(-22.0, min(12.0, y_movement))


def scale_person(person_index: int, current_time: float, beat: dict[str, float | int]) -> float:
    """Apply a restrained whole-person pulse while respecting safe scale limits."""
    phase = current_time * 1.45 + person_index * 0.4
    scale = 1.0 + 0.025 * (0.5 + 0.5 * math.sin(phase))
    scale += float(beat["accent"]) * 0.012
    return max(0.95, min(1.06, scale))


def apply_person_transform(
    person_image: Image.Image,
    person_index: int,
    current_time: float,
    beat: dict[str, float | int],
) -> tuple[Image.Image, float, float]:
    """Rotate and resize a complete transparent person layer without facial warping."""
    rotation = rotate_person(person_index, current_time, beat)
    scale = scale_person(person_index, current_time, beat)
    target_size = (
        max(1, int(person_image.width * scale)),
        max(1, int(person_image.height * scale)),
    )
    transformed = person_image.resize(target_size, Image.Resampling.LANCZOS)
    transformed = transformed.rotate(rotation, resample=Image.Resampling.BICUBIC, expand=True)
    group = group_dance(current_time, beat)
    move_x, move_y = move_person(person_index, current_time, beat, group)
    return transformed, move_x, move_y


def _draw_flower(
    draw: ImageDraw.ImageDraw,
    center_x: float,
    center_y: float,
    radius: float,
    color: tuple[int, int, int, int],
    rotation: float = 0.0,
) -> None:
    """Draw one small layered flower using ellipses and a center circle."""
    for petal_index in range(8):
        angle = rotation + petal_index * math.tau / 8
        petal_x = center_x + math.cos(angle) * radius * 0.52
        petal_y = center_y + math.sin(angle) * radius * 0.52
        petal_radius = radius * 0.38
        draw.ellipse(
            (petal_x - petal_radius, petal_y - petal_radius,
             petal_x + petal_radius, petal_y + petal_radius),
            fill=color,
        )
    draw.ellipse(
        (center_x - radius * 0.25, center_y - radius * 0.25,
         center_x + radius * 0.25, center_y + radius * 0.25),
        fill=(255, 235, 156, 245),
    )


def create_bathukamma(frame: Image.Image, current_time: float) -> None:
    """Draw a colorful, layered Bathukamma flower arrangement at the stage edge."""
    draw = ImageDraw.Draw(frame, "RGBA")
    center_x = frame.width * 0.5 + math.sin(current_time * 0.45) * 3
    base_y = frame.height - 54
    draw.ellipse((center_x - 122, base_y - 28, center_x + 122, base_y + 10), fill=(10, 25, 26, 110))

    tiers = [
        (116, 20, 12, 0),
        (98, 19, 11, 1),
        (80, 17, 10, 2),
        (62, 15, 9, 3),
        (45, 13, 8, 4),
        (29, 11, 7, 0),
        (14, 8, 5, 1),
    ]
    for tier_index, (radius_x, radius_y, flower_radius, color_offset) in enumerate(tiers):
        center_y = base_y - 19 - tier_index * 18
        flower_count = max(7, radius_x // 9)
        for flower_index in range(flower_count):
            angle = math.tau * flower_index / flower_count
            flower_x = center_x + math.cos(angle) * radius_x
            flower_y = center_y + math.sin(angle) * radius_y
            color = FLOWER_COLORS[(flower_index + color_offset) % len(FLOWER_COLORS)]
            _draw_flower(draw, flower_x, flower_y, flower_radius, color, angle)
        draw.ellipse(
            (center_x - radius_x, center_y - radius_y * 0.55,
             center_x + radius_x, center_y + radius_y * 0.55),
            outline=(255, 223, 137, 115),
            width=2,
        )


def create_background(
    width: int,
    height: int,
    source_image: Image.Image,
    current_time: float,
) -> Image.Image:
    """Build and reuse a warm, patterned stage background inspired by the source colors."""
    cache_key = (width, height)
    if cache_key not in _BACKGROUND_CACHE:
        sample = ImageStat.Stat(source_image.convert("RGB").resize((8, 8))).mean
        source_tint = np.asarray(sample, dtype=np.float32)
        top_color = np.asarray((20, 69, 60), dtype=np.float32) * 0.82 + source_tint * 0.18
        bottom_color = np.asarray((103, 53, 60), dtype=np.float32)
        vertical = np.linspace(0.0, 1.0, height, dtype=np.float32)[:, None, None]
        gradient = top_color[None, None, :] * (1.0 - vertical) + bottom_color[None, None, :] * vertical
        pixels = np.broadcast_to(gradient, (height, width, 3)).astype(np.uint8).copy()
        background = Image.fromarray(pixels).convert("RGBA")
        draw = ImageDraw.Draw(background, "RGBA")
        for row in range(0, height, 42):
            draw.line((0, row, width, row), fill=(255, 226, 166, 15), width=1)
        for side_x in (38, width - 38):
            for circle_y in range(72, height - 95, 92):
                draw.ellipse(
                    (side_x - 19, circle_y - 19, side_x + 19, circle_y + 19),
                    outline=(248, 206, 129, 52),
                    width=2,
                )
                draw.ellipse(
                    (side_x - 4, circle_y - 4, side_x + 4, circle_y + 4),
                    fill=(248, 206, 129, 76),
                )
        draw.rectangle((0, height - 57, width, height), fill=(19, 42, 37, 140))
        draw.line((0, height - 58, width, height - 58), fill=(242, 183, 54, 140), width=2)
        _BACKGROUND_CACHE[cache_key] = background

    frame = _BACKGROUND_CACHE[cache_key].copy()
    draw = ImageDraw.Draw(frame, "RGBA")
    drift = math.sin(current_time * 0.5) * 8
    for index in range(10):
        x = (index * 139 + int(current_time * 8)) % width
        y = 60 + (index * 71) % max(1, height - 160)
        radius = 2 + (index % 3)
        draw.ellipse((x + drift - radius, y - radius, x + drift + radius, y + radius), fill=(255, 232, 174, 76))
    return frame


def make_particles(count: int = 46) -> list[dict[str, float | str]]:
    """Create repeatable drifting petal and soft-light particle properties."""
    generator = random.Random(42)
    particles = []
    for index in range(count):
        particles.append({
            "x": generator.uniform(20, WIDTH - 20),
            "y": generator.uniform(-HEIGHT, HEIGHT),
            "speed": generator.uniform(13, 34),
            "size": generator.uniform(2.5, 7.0),
            "rotation": generator.uniform(0, math.tau),
            "drift": generator.uniform(5, 22),
            "phase": generator.uniform(0, math.tau),
            "type": "light" if index % 4 == 0 else "petal",
        })
    return particles


def add_flower_particles(
    frame: Image.Image,
    particles: list[dict[str, float | str]],
    current_time: float,
) -> None:
    """Composite subtle, deterministic flower petals and light particles over a frame."""
    overlay = Image.new("RGBA", frame.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay, "RGBA")
    for index, particle in enumerate(particles):
        size = float(particle["size"])
        phase = float(particle["phase"])
        x = (float(particle["x"]) + math.sin(current_time * 0.7 + phase) * float(particle["drift"])) % frame.width
        y = (float(particle["y"]) + float(particle["speed"]) * current_time) % frame.height
        if particle["type"] == "light":
            radius = size * 0.48
            draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=(255, 241, 192, 82))
        else:
            color = FLOWER_COLORS[index % len(FLOWER_COLORS)]
            angle = float(particle["rotation"]) + current_time * 0.35
            points = []
            for point_index in range(5):
                point_angle = angle + point_index * math.tau / 5
                point_radius = size if point_index % 2 == 0 else size * 0.48
                points.append((x + math.cos(point_angle) * point_radius, y + math.sin(point_angle) * point_radius))
            draw.polygon(points, fill=(color[0], color[1], color[2], 150))
    frame.alpha_composite(overlay)


def composite_person(frame: Image.Image, person: Image.Image, center_x: float, bottom_y: float) -> None:
    """Alpha-composite a transformed cutout while safely clipping only off-canvas padding."""
    left = int(round(center_x - person.width / 2))
    top = int(round(bottom_y - person.height))
    right = left + person.width
    bottom = top + person.height
    dest_left = max(0, left)
    dest_top = max(0, top)
    dest_right = min(frame.width, right)
    dest_bottom = min(frame.height, bottom)
    if dest_left >= dest_right or dest_top >= dest_bottom:
        return
    crop = person.crop((dest_left - left, dest_top - top, dest_right - left, dest_bottom - top))
    frame.alpha_composite(crop, (dest_left, dest_top))


def _transform_hand_point(
    point: tuple[float, float],
    layer: dict[str, Any],
    person_index: int,
    current_time: float,
    beat: dict[str, float | int],
    center_x: float,
    bottom_y: float,
    transformed: Image.Image,
) -> tuple[float, float]:
    """Apply the same resize, rotation, and translation as the person's wrist pixel."""
    source_x, source_y = point
    crop_x, crop_y = layer["crop_origin"]
    source_width, source_height = layer["source_crop_size"]
    prepared = layer["image"]
    scale = scale_person(person_index, current_time, beat)
    resized_width = max(1, int(prepared.width * scale))
    resized_height = max(1, int(prepared.height * scale))
    point_x = (source_x - crop_x) * prepared.width / source_width
    point_y = (source_y - crop_y) * prepared.height / source_height
    point_x = point_x * resized_width / prepared.width - resized_width / 2
    point_y = point_y * resized_height / prepared.height - resized_height / 2

    angle = math.radians(rotate_person(person_index, current_time, beat))
    rotated_x = math.cos(angle) * point_x + math.sin(angle) * point_y
    rotated_y = -math.sin(angle) * point_x + math.cos(angle) * point_y
    return center_x + rotated_x, bottom_y - transformed.height / 2 + rotated_y


def add_hand_connections(
    frame: Image.Image,
    hands: list[dict[str, tuple[float, float]]],
) -> None:
    """Visually join each adjacent pair at their detected wrist positions."""
    draw = ImageDraw.Draw(frame, "RGBA")
    for left_dancer, right_dancer in zip(hands, hands[1:]):
        start = left_dancer.get("screen_right")
        end = right_dancer.get("screen_left")
        if start is None or end is None:
            continue
        distance = math.hypot(end[0] - start[0], end[1] - start[1])
        if distance > 175:
            continue

        control_x = (start[0] + end[0]) / 2
        control_y = (start[1] + end[1]) / 2 - min(12.0, distance * 0.08)
        curve = []
        for step in range(13):
            amount = step / 12
            inverse = 1 - amount
            x = inverse * inverse * start[0] + 2 * inverse * amount * control_x + amount * amount * end[0]
            y = inverse * inverse * start[1] + 2 * inverse * amount * control_y + amount * amount * end[1]
            curve.append((x, y))
        draw.line(curve, fill=(36, 41, 34, 235), width=24, joint="curve")
        draw.line(curve, fill=(255, 218, 135, 255), width=14, joint="curve")
        middle_x, middle_y = curve[len(curve) // 2]
        draw.ellipse(
            (middle_x - 12, middle_y - 12, middle_x + 12, middle_y + 12),
            fill=(242, 183, 54, 255),
            outline=(255, 247, 222, 255),
            width=1,
        )


def _font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    """Find a readable installed font and fall back to Pillow's built-in font."""
    windows_fonts = Path(os.environ.get("WINDIR", r"C:\Windows")) / "Fonts"
    candidates = [
        windows_fonts / ("arialbd.ttf" if bold else "arial.ttf"),
        windows_fonts / ("georgiab.ttf" if bold else "georgia.ttf"),
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
    ]
    for candidate in candidates:
        if candidate.is_file():
            try:
                return ImageFont.truetype(str(candidate), size=size)
            except OSError:
                continue
    return ImageFont.load_default()


def draw_title(frame: Image.Image, current_time: float) -> None:
    """Show the requested short title card with a simple fade and scale."""
    if current_time >= 1.5:
        return
    fade_in = smooth_step(min(1.0, current_time / 0.25))
    fade_out = 1.0 if current_time <= 1.08 else 1.0 - smooth_step((current_time - 1.08) / 0.42)
    opacity = int(230 * fade_in * fade_out)
    scale = 0.96 + 0.04 * smooth_step(min(1.0, current_time / 0.8))
    title_font = _font(int(54 * scale), bold=True)
    subtitle_font = _font(20, bold=False)
    card = Image.new("RGBA", (frame.width, 150), (0, 0, 0, 0))
    draw = ImageDraw.Draw(card, "RGBA")
    title = "BATHUKAMMA"
    subtitle = "TELANGANA CULTURAL DANCE"
    title_box = draw.textbbox((0, 0), title, font=title_font, stroke_width=0)
    subtitle_box = draw.textbbox((0, 0), subtitle, font=subtitle_font)
    title_x = (frame.width - (title_box[2] - title_box[0])) / 2
    subtitle_x = (frame.width - (subtitle_box[2] - subtitle_box[0])) / 2
    draw.text((title_x + 2, 24), title, font=title_font, fill=(0, 0, 0, int(opacity * 0.45)))
    draw.text((title_x, 22), title, font=title_font, fill=(255, 247, 222, opacity))
    draw.text((subtitle_x, 91), subtitle, font=subtitle_font, fill=(255, 216, 132, opacity))
    frame.alpha_composite(card)


def apply_camera_motion(frame: Image.Image, current_time: float) -> Image.Image:
    """Add a tiny zoom-and-pan move, leaving safe margins around the dancers."""
    zoom = 1.0 + 0.025 * (0.5 - 0.5 * math.cos(math.tau * current_time / DURATION))
    scaled_width = int(frame.width * zoom)
    scaled_height = int(frame.height * zoom)
    scaled = frame.resize((scaled_width, scaled_height), Image.Resampling.LANCZOS)
    max_left = max(0, scaled_width - frame.width)
    max_top = max(0, scaled_height - frame.height)
    drift_x = int(max_left * (0.5 + 0.3 * math.sin(current_time * 0.35)))
    drift_y = int(max_top * (0.5 + 0.3 * math.cos(current_time * 0.3)))
    return scaled.crop((drift_x, drift_y, drift_x + frame.width, drift_y + frame.height))


def render_video(
    person_layers: list[dict[str, Any]],
    source_image: Image.Image,
    audio_analysis: dict[str, Any],
    duration: float = DURATION,
) -> Path:
    """Render the animated dancers to a silent H.264 MP4 using imageio's FFmpeg."""
    total_frames = max(1, int(round(duration * FPS)))
    beat_times = [float(value) for value in audio_analysis.get("beats", [])]
    median_height = float(np.median([layer["image"].height for layer in person_layers]))
    target_height = HEIGHT * 0.77
    prepared_layers = []
    for layer in person_layers:
        image = layer["image"]
        source_crop_size = image.size
        scale = min(target_height / max(1, median_height), WIDTH * 0.19 / max(1, image.width))
        size = (max(1, int(image.width * scale)), max(1, int(image.height * scale)))
        prepared_layers.append({
            "image": image.resize(size, Image.Resampling.LANCZOS),
            "center_x": float(layer["center_x"]) * WIDTH,
            "source_crop_size": source_crop_size,
            "crop_origin": layer.get("crop_origin", (0, 0)),
            "hand_points": layer.get("hand_points", {}),
        })

    particles = make_particles()
    GENERATED_DIR.mkdir(parents=True, exist_ok=True)
    print("Rendering: 0%")
    with imageio.get_writer(
        str(SILENT_VIDEO_PATH),
        fps=FPS,
        codec="libx264",
        quality=8,
        pixelformat="yuv420p",
        macro_block_size=1,
        ffmpeg_params=["-preset", "medium", "-crf", "20", "-an"],
    ) as writer:
        progress_step = max(1, total_frames // 20)
        for frame_number in range(total_frames):
            current_time = frame_number / FPS
            frame = create_background(WIDTH, HEIGHT, source_image, current_time)
            beat = _beat_state(current_time, beat_times)
            dancer_hands = []
            arc_offsets = (-54.0, -20.0, 0.0, -20.0, -54.0)
            hand_arc_offsets = (10.0, 5.0, 0.0, 5.0, 10.0)
            for person_index, layer in enumerate(prepared_layers):
                transformed, move_x, move_y = apply_person_transform(
                    layer["image"], person_index, current_time, beat
                )
                center_x = layer["center_x"] + move_x
                center_x = max(
                    transformed.width / 2 + 8,
                    min(WIDTH - transformed.width / 2 - 8, center_x),
                )
                bottom_y = HEIGHT - 12 + arc_offsets[person_index] + move_y
                if layer["hand_points"]:
                    wrist_offsets = [
                        _transform_hand_point(
                            point,
                            layer,
                            person_index,
                            current_time,
                            beat,
                            center_x,
                            0.0,
                            transformed,
                        )[1]
                        for point in layer["hand_points"].values()
                    ]
                    if wrist_offsets:
                        target_wrist_y = HEIGHT * 0.48 + hand_arc_offsets[person_index]
                        bottom_y = target_wrist_y - sum(wrist_offsets) / len(wrist_offsets) + move_y
                bottom_y = max(transformed.height + 8, min(HEIGHT - 2, bottom_y))
                composite_person(frame, transformed, center_x, bottom_y)

                visible_hands = {}
                for side, source_point in layer["hand_points"].items():
                    visible_hands[side] = _transform_hand_point(
                        source_point,
                        layer,
                        person_index,
                        current_time,
                        beat,
                        center_x,
                        bottom_y,
                        transformed,
                    )
                dancer_hands.append(visible_hands)

            create_bathukamma(frame, current_time)
            add_hand_connections(frame, dancer_hands)
            add_flower_particles(frame, particles, current_time)
            draw_title(frame, current_time)
            frame = apply_camera_motion(frame, current_time)
            writer.append_data(np.asarray(frame.convert("RGB"), dtype=np.uint8))

            if (frame_number + 1) % progress_step == 0 or frame_number + 1 == total_frames:
                percent = int((frame_number + 1) / total_frames * 100)
                print(f"Rendering: {percent}%")
    return SILENT_VIDEO_PATH


def create_final_video(silent_video_path: Path, audio_path: Path, duration: float) -> Path:
    """Mux the song into the landscape render and keep the finished MP4 in output/."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    command = [
        imageio_ffmpeg.get_ffmpeg_exe(),
        "-y",
        "-i", str(silent_video_path),
        "-stream_loop", "-1",
        "-i", str(audio_path),
        "-map", "0:v:0",
        "-map", "1:a:0",
        "-t", str(duration),
        "-c:v", "copy",
        "-c:a", "aac",
        "-b:a", "192k",
        "-movflags", "+faststart",
        str(OUTPUT_VIDEO_PATH),
    ]
    try:
        subprocess.run(command, check=True, capture_output=True, text=True)
    except (OSError, subprocess.CalledProcessError) as error:
        details = error.stderr.strip() if isinstance(error, subprocess.CalledProcessError) and error.stderr else str(error)
        raise RuntimeError(f"Could not create the final video: {details}") from error
    return OUTPUT_VIDEO_PATH


def open_video(video_path: Path) -> bool:
    """Open the finished video with the operating system's default video player."""
    try:
        if sys.platform == "win32":
            os.startfile(str(video_path))
        elif sys.platform == "darwin":
            subprocess.Popen(["open", str(video_path)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        else:
            subprocess.Popen(["xdg-open", str(video_path)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except (AttributeError, OSError):
        return False
    return True


def parse_arguments() -> argparse.Namespace:
    """Read optional preview-duration settings; normal use remains simply python main.py."""
    parser = argparse.ArgumentParser(description="Generate and play a 35-second Bathukamma dance preview.")
    parser.add_argument(
        "--test-seconds",
        type=float,
        default=None,
        help="Render and play a shorter test preview instead of the default 35 seconds.",
    )
    arguments = parser.parse_args()
    if arguments.test_seconds is not None and not 0.5 <= arguments.test_seconds <= DURATION:
        parser.error("--test-seconds must be between 0.5 and 35.")
    return arguments


def main() -> int:
    """Run segmentation, beat analysis, rendering, and open the saved video."""
    arguments = parse_arguments()
    render_duration = arguments.test_seconds or DURATION

    print("=" * 60)
    print("        BATHUKAMMA GROUP DANCE VIDEO GENERATOR")
    print("=" * 60)
    ensure_directories()

    try:
        print("\n[1/7] Checking input files...")
        image_path, audio_path = check_input_files()

        print("[2/7] Loading source image...")
        with Image.open(image_path) as opened_image:
            source_image = opened_image.convert("RGB")
        print(f"Loaded image: {source_image.width} x {source_image.height}")

        print("[3/7] Detecting five dancers...")
        person_layers = extract_people(image_path)

        print("[4/7] Analyzing Bathukamma song...")
        audio_analysis = analyze_audio(audio_path)
        audio_analysis["render_duration"] = render_duration
        ANALYSIS_PATH.write_text(json.dumps(audio_analysis, indent=2), encoding="utf-8")

        print("[5/7] Creating dance animation...")
        print(f"[6/7] Rendering {render_duration:g}-second video at {WIDTH}x{HEIGHT}, {FPS} FPS...")
        try:
            silent_video_path = render_video(person_layers, source_image, audio_analysis, render_duration)
            print("[7/7] Saving the landscape video with audio...")
            final_video_path = create_final_video(silent_video_path, audio_path, render_duration)
            if open_video(final_video_path):
                print("Opened the finished video in your default video player.")
            else:
                print("Could not open a video player automatically; open the saved MP4 manually.")
        finally:
            SILENT_VIDEO_PATH.unlink(missing_ok=True)

        print("\n" + "=" * 60)
        print("BATHUKAMMA VIDEO READY!")
        print("=" * 60)
        print(f"Duration:   {render_duration:g} seconds")
        print(f"Resolution: {WIDTH} x {HEIGHT}")
        print(f"FPS:        {FPS}")
        print("Audio:      Supplied Bathukamma song")
        print(f"Saved:      {OUTPUT_VIDEO_PATH.relative_to(ROOT_DIR)}")
        print(f"Dancers:    {len(person_layers)}")
        print("=" * 60)
        return 0
    except KeyboardInterrupt:
        print("\nVideo generation stopped.")
        return 130
    except SegmentationError as error:
        print(f"\n{error}")
        return 1
    except (FileNotFoundError, OSError, RuntimeError, ValueError) as error:
        print(f"\nERROR: {error}")
        return 1


if __name__ == "__main__":
    sys.exit(main())