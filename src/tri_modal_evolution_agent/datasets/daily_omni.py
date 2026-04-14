from __future__ import annotations

from pathlib import Path

from .common import build_mcq_prompt, infer_gold_choice, load_json, normalize_options
from ..types import BenchmarkSample


def load_daily_omni_samples(dataset_config: dict, model_config: dict) -> list[BenchmarkSample]:
    paths = dataset_config["paths"]
    prompts = dataset_config["prompts"]
    generation = dict(model_config.get("generation_defaults", {}))
    generation.update(dataset_config.get("generation_overrides", {}))

    dataset_root = Path(paths["dataset_root"]).expanduser().resolve()
    qa_file = Path(paths.get("qa_file") or dataset_root / "qa.json").expanduser().resolve()
    rows = load_json(qa_file)

    samples: list[BenchmarkSample] = []
    for index, row in enumerate(rows):
        options = normalize_options(row["Choice"])
        correct_answer = infer_gold_choice(str(row["Answer"]), options)
        video_id = str(row["video_id"])
        video_dir = dataset_root / "Videos" / video_id
        video_path = video_dir / f"{video_id}_video.mp4"
        audio_path = video_dir / f"{video_id}_audio.wav"
        prompt = build_mcq_prompt(
            str(row["Question"]),
            options,
            prompts["modality_hint"],
            prompts["response_suffix"],
        )
        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": prompts["system_prompt"]}],
            },
            {
                "role": "user",
                "content": [
                    {"type": "video", "video": str(video_path)},
                    {"type": "audio", "audio": str(audio_path)},
                    {"type": "text", "text": prompt},
                ],
            },
        ]
        samples.append(
            BenchmarkSample(
                sample_id=f"{index:04d}",
                dataset_key=str(dataset_config["dataset_key"]),
                question=str(row["Question"]),
                options=options,
                correct_answer=correct_answer,
                messages=messages,
                media_paths={
                    "video_path": str(video_path),
                    "audio_path": str(audio_path),
                },
                metadata={
                    "index": index,
                    "video_id": video_id,
                    "task_type": row.get("Type", ""),
                    "content_parent_category": row.get("content_parent_category", ""),
                    "content_fine_category": row.get("content_fine_category", ""),
                    "video_category": row.get("video_category", ""),
                    "video_duration": row.get("video_duration", ""),
                    "qa_file": str(qa_file),
                },
                generation=generation,
            )
        )
    return samples
