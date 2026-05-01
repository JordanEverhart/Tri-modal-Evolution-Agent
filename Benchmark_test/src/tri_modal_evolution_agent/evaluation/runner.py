from __future__ import annotations

import json
import time
import traceback
from pathlib import Path
from typing import Iterable

from ..datasets.common import build_index_to_answer, parse_choice_response
from ..engine.api_client import LocalOmniApiClient
from ..types import BenchmarkSample
from .reporting import dump_json, load_done_sample_ids, load_jsonl, summarize_results, write_errors


def _result_row(sample: BenchmarkSample, *, model_key: str, response_text: str, parsed_response: str) -> dict:
    index2ans = build_index_to_answer(sample.options)
    row = {
        "sample_id": sample.sample_id,
        "dataset_key": sample.dataset_key,
        "model_key": model_key,
        "question": sample.question,
        "options": sample.options,
        "correct_answer": sample.correct_answer,
        "correct_answer_text": index2ans.get(sample.correct_answer),
        "raw_response": response_text,
        "parsed_response": parsed_response,
        "is_correct": parsed_response == sample.correct_answer,
        "media_paths": sample.media_paths,
    }
    row.update(sample.metadata)
    return row


def _error_row(sample: BenchmarkSample, *, model_key: str, error_message: str) -> dict:
    index2ans = build_index_to_answer(sample.options)
    row = {
        "sample_id": sample.sample_id,
        "dataset_key": sample.dataset_key,
        "model_key": model_key,
        "question": sample.question,
        "options": sample.options,
        "correct_answer": sample.correct_answer,
        "correct_answer_text": index2ans.get(sample.correct_answer),
        "raw_response": "",
        "parsed_response": "N/A",
        "is_correct": False,
        "error": error_message,
        "media_paths": sample.media_paths,
    }
    row.update(sample.metadata)
    return row


def evaluate_samples(
    *,
    samples: Iterable[BenchmarkSample],
    client: LocalOmniApiClient,
    model_key: str,
    results_path: str | Path,
    summary_path: str | Path,
    errors_path: str | Path,
    group_fields: list[str],
    resume: bool = False,
) -> dict[str, Path]:
    results_path = Path(results_path).expanduser().resolve()
    results_path.parent.mkdir(parents=True, exist_ok=True)
    done_sample_ids = load_done_sample_ids(results_path) if resume else set()
    started_at = time.time()

    write_mode = "a" if resume else "w"
    with results_path.open(write_mode, encoding="utf-8") as handle:
        processed = 0
        for sample in samples:
            if sample.sample_id in done_sample_ids:
                continue
            processed += 1
            try:
                payload = client.generate(messages=sample.messages, generation=sample.generation)
                response_text = str(payload.get("text", "")).strip()
                parsed_response = parse_choice_response(response_text, sample.options)
                row = _result_row(
                    sample,
                    model_key=model_key,
                    response_text=response_text,
                    parsed_response=parsed_response,
                )
            except RuntimeError as exc:
                traceback.print_exc()
                row = _error_row(
                    sample,
                    model_key=model_key,
                    error_message=str(exc),
                )
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            handle.flush()
            if processed == 1 or processed % 25 == 0:
                elapsed = time.time() - started_at
                print(
                    f"[eval] dataset={sample.dataset_key} processed={processed} "
                    f"sample_id={sample.sample_id} elapsed_sec={elapsed:.1f}",
                    flush=True,
                )

    results = load_jsonl(results_path)
    if not results:
        raise RuntimeError(f"No results were written to {results_path}")

    dataset_key = results[0]["dataset_key"]
    summary = summarize_results(
        results,
        dataset_key=dataset_key,
        model_key=model_key,
        group_fields=group_fields,
    )
    dump_json(summary_path, summary)
    write_errors(errors_path, results)
    print(
        f"[eval] dataset={dataset_key} finished total={len(results)} accuracy={summary['accuracy']}",
        flush=True,
    )
    return {
        "results": results_path,
        "summary": Path(summary_path).expanduser().resolve(),
        "errors": Path(errors_path).expanduser().resolve(),
    }
