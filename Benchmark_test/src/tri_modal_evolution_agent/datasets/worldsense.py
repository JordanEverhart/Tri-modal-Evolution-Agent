from __future__ import annotations

from pathlib import Path

from .common import build_mcq_prompt, infer_gold_choice, load_json, normalize_options
from ..types import BenchmarkSample


def _resolve_worldsense_video_path(dataset_root: Path, video_id: str) -> Path:
    candidates = [
        dataset_root / "videos" / f"{video_id}.mp4",
        dataset_root / "videos" / f"{video_id}.mkv",
        dataset_root / f"{video_id}.mp4",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Could not resolve WorldSense video for video_id={video_id}")


def _resolve_worldsense_audio_path(dataset_root: Path, video_id: str) -> Path:
    candidates = [
        dataset_root / "audios" / f"{video_id}.wav",
        dataset_root / "audios" / f"{video_id}.mp3",
        dataset_root / f"{video_id}.wav",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Could not resolve WorldSense audio for video_id={video_id}")


def load_worldsense_samples(dataset_config: dict, model_config: dict) -> list[BenchmarkSample]:
    paths = dataset_config["paths"]
    prompts = dataset_config["prompts"]
    generation = dict(model_config.get("generation_defaults", {}))
    generation.update(dataset_config.get("generation_overrides", {}))

    dataset_root = Path(paths["dataset_root"]).expanduser().resolve()
    qa_file = Path(paths.get("qa_file") or dataset_root / "worldsense_qa.json").expanduser().resolve()
    rows = load_json(qa_file)

    samples: list[BenchmarkSample] = []
    for video_id, entry in rows.items():
        try:
            video_path = _resolve_worldsense_video_path(dataset_root, str(video_id))
            audio_path = _resolve_worldsense_audio_path(dataset_root, str(video_id))
        except FileNotFoundError:
            continue

        subtitle_path = dataset_root / "subtitles" / f"{video_id}.srt"
        shared_metadata = {
            "video_id": str(video_id),
            "video_duration": entry.get("video_duration", ""),
            "duration_bucket": entry.get("duration", ""),
            "domain": entry.get("domain", ""),
            "sub_category": entry.get("sub_category", ""),
            "audio_class": entry.get("audio_class", []),
            "video_caption": entry.get("video_caption", ""),
            "qa_file": str(qa_file),
        }

        for task_key, task in entry.items():
            if not str(task_key).startswith("task"):
                continue
            options = normalize_options(task["candidates"])
            correct_answer = infer_gold_choice(str(task["answer"]), options)
            prompt = build_mcq_prompt(
                str(task["question"]),
                options,
                prompts["modality_hint"],
                prompts["response_suffix"],
            )
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompts["system_prompt"]},
                        {"type": "video", "video": str(video_path), "fps": float(dataset_config["media"]["video_fps"])},
                        {"type": "audio", "audio": str(audio_path)},
                        {"type": "text", "text": prompts["media_prompt"]},
                        {"type": "text", "text": prompt},
                    ],
                }
            ]
            samples.append(
                BenchmarkSample(
                    sample_id=f"{video_id}__{task_key}",
                    dataset_key=str(dataset_config["dataset_key"]),
                    question=str(task["question"]),
                    options=options,
                    correct_answer=correct_answer,
                    messages=messages,
                    media_paths={
                        "video_path": str(video_path),
                        "audio_path": str(audio_path),
                        "subtitle_path": str(subtitle_path),
                    },
                    metadata={
                        **shared_metadata,
                        "task_key": str(task_key),
                        "task_domain": task.get("task_domain", ""),
                        "task_type": task.get("task_type", ""),
                    },
                    generation=generation,
                )
            )
    return samples
