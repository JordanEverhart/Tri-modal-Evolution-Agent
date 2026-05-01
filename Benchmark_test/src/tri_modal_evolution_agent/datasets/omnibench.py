from __future__ import annotations

from pathlib import Path

from .common import build_mcq_prompt, infer_gold_choice, load_jsonl, normalize_options
from ..types import BenchmarkSample


def load_omnibench_samples(dataset_config: dict, model_config: dict) -> list[BenchmarkSample]:
    paths = dataset_config["paths"]
    prompts = dataset_config["prompts"]
    generation = dict(model_config.get("generation_defaults", {}))
    generation.update(dataset_config.get("generation_overrides", {}))

    repo_root = Path(paths["repo_root"]).expanduser().resolve()
    dataset_file = Path(paths.get("dataset_file") or repo_root / "dataset" / "batch-5_1142_20240817.jsonl").expanduser().resolve()
    mm_root = Path(paths.get("mm_root") or repo_root / "mm_data").expanduser().resolve()

    samples: list[BenchmarkSample] = []
    for row in load_jsonl(dataset_file):
        options = normalize_options(row["options"])
        correct_answer = infer_gold_choice(str(row["answer"]), options)
        image_path = mm_root / "image" / str(row["image_path"])
        audio_path = mm_root / "audio" / str(row["audio_path"])
        prompt = build_mcq_prompt(
            str(row["question"]),
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
                    {"type": "image", "image": str(image_path)},
                    {"type": "audio", "audio": str(audio_path)},
                    {"type": "text", "text": prompt},
                ],
            },
        ]
        samples.append(
            BenchmarkSample(
                sample_id=str(row["index"]),
                dataset_key=str(dataset_config["dataset_key"]),
                question=str(row["question"]),
                options=options,
                correct_answer=correct_answer,
                messages=messages,
                media_paths={
                    "image_path": str(image_path),
                    "audio_path": str(audio_path),
                },
                metadata={
                    "index": int(row["index"]),
                    "task_type": row.get("task type", row.get("task_type", "")),
                    "audio_type": row.get("audio type", row.get("audio_type", "")),
                    "dataset_file": str(dataset_file),
                },
                generation=generation,
            )
        )
    return samples
