from __future__ import annotations

import argparse
import gc
import time
import traceback
from threading import Lock
from typing import Any

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from ..config.loader import load_yaml_config


class GenerateRequest(BaseModel):
    messages: list[dict[str, Any]]
    generation: dict[str, Any] = Field(default_factory=dict)


class Qwen3OmniRunner:
    def __init__(self, model_config: dict):
        self.model_config = model_config
        self.runner = self._build_runner()

    @staticmethod
    def _clear_cuda_memory(torch: Any) -> None:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    @staticmethod
    def _is_oom_error(exc: Exception) -> bool:
        return "out of memory" in str(exc).lower()

    @staticmethod
    def _resolve_dtype(raw_dtype: Any, torch: Any) -> Any:
        if raw_dtype in (None, "", "auto"):
            return "auto"
        if isinstance(raw_dtype, str):
            normalized = raw_dtype.strip().lower()
            mapping = {
                "fp16": torch.float16,
                "float16": torch.float16,
                "half": torch.float16,
                "bf16": torch.bfloat16,
                "bfloat16": torch.bfloat16,
                "fp32": torch.float32,
                "float32": torch.float32,
            }
            if normalized in mapping:
                return mapping[normalized]
        return raw_dtype

    @staticmethod
    def _resolve_device_map(raw_device_map: Any, server_cfg: dict, torch: Any) -> Any:
        if raw_device_map in (None, "", "auto"):
            return "auto"
        if isinstance(raw_device_map, str):
            normalized = raw_device_map.strip().lower()
            if normalized in {"single_gpu", "single-gpu", "single"}:
                if not torch.cuda.is_available():
                    raise RuntimeError("single_gpu device_map requires a visible CUDA device.")
                cuda_index = int(server_cfg.get("cuda_device", 0))
                return {"": f"cuda:{cuda_index}"}
            if normalized.startswith("cuda:"):
                return {"": normalized}
        return raw_device_map

    @staticmethod
    def _resolve_execution_device(model: Any, torch: Any) -> Any:
        hf_device_map = getattr(model, "hf_device_map", None)
        if isinstance(hf_device_map, dict):
            for target in hf_device_map.values():
                if isinstance(target, int):
                    return torch.device(f"cuda:{target}")
                if isinstance(target, str):
                    if target in {"cpu", "disk", "meta"}:
                        continue
                    return torch.device(target)
                if isinstance(target, torch.device):
                    if target.type not in {"cpu", "meta"}:
                        return target

        model_device = getattr(model, "device", None)
        if isinstance(model_device, torch.device) and model_device.type not in {"cpu", "meta"}:
            return model_device

        for parameter in model.parameters():
            if parameter.device.type not in {"cpu", "meta"}:
                return parameter.device
            if parameter.device.type == "cpu":
                return parameter.device

        if torch.cuda.is_available():
            return torch.device("cuda:0")
        return torch.device("cpu")

    @staticmethod
    def _resolve_model_dtype(model: Any, torch: Any) -> Any:
        for parameter in model.parameters():
            if parameter.device.type != "meta":
                return parameter.dtype
        if torch.cuda.is_available():
            return torch.bfloat16
        return torch.float32

    def _build_runner(self) -> Any:
        try:
            import torch
            from qwen_omni_utils import process_mm_info
            from transformers import Qwen3OmniMoeForConditionalGeneration, Qwen3OmniMoeProcessor
        except ImportError as exc:
            raise RuntimeError(
                "Missing Qwen3-Omni runtime dependencies. Please use the qwen3omni environment."
            ) from exc

        model_path = self.model_config["model"]["path"]
        server_cfg = self.model_config.get("server", {})
        resolved_dtype = self._resolve_dtype(server_cfg.get("dtype", "auto"), torch)
        resolved_device_map = self._resolve_device_map(server_cfg.get("device_map", "auto"), server_cfg, torch)
        kwargs: dict[str, Any] = {
            "dtype": resolved_dtype,
            "device_map": resolved_device_map,
            "low_cpu_mem_usage": True,
        }
        if server_cfg.get("flash_attn2", False):
            kwargs["attn_implementation"] = "flash_attention_2"

        model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(model_path, **kwargs)
        if server_cfg.get("disable_talker", False):
            model.disable_talker()
        model.eval()
        processor = Qwen3OmniMoeProcessor.from_pretrained(model_path)
        execution_device = self._resolve_execution_device(model, torch)
        model_dtype = self._resolve_model_dtype(model, torch)
        device_map_targets = sorted({str(target) for target in getattr(model, "hf_device_map", {}).values()})
        print(
            "[serve] "
            f"resolved_device_map={resolved_device_map} "
            f"execution_device={execution_device} "
            f"model_dtype={model_dtype} "
            f"hf_device_map_targets={device_map_targets}",
            flush=True,
        )

        return {
            "torch": torch,
            "process_mm_info": process_mm_info,
            "model": model,
            "processor": processor,
            "execution_device": execution_device,
            "model_dtype": model_dtype,
        }

    def clear_cached_memory(self) -> None:
        self._clear_cuda_memory(self.runner["torch"])

    def generate_text(self, messages: list[dict[str, Any]], generation: dict[str, Any]) -> str:
        torch = self.runner["torch"]
        process_mm_info = self.runner["process_mm_info"]
        model = self.runner["model"]
        processor = self.runner["processor"]
        execution_device = self.runner["execution_device"]
        model_dtype = self.runner["model_dtype"]
        text = None
        audios = None
        images = None
        videos = None
        inputs = None
        generated = None
        generated_ids = None
        sequences = None
        new_tokens = None

        use_audio_in_video = bool(generation.get("use_audio_in_video", True))
        max_new_tokens = int(generation.get("max_new_tokens", 64))
        do_sample = bool(generation.get("do_sample", False))
        temperature = float(generation.get("temperature", 0.0))
        top_p = float(generation.get("top_p", 1.0))
        top_k = int(generation.get("top_k", 20))

        try:
            text = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            audios, images, videos = process_mm_info(messages, use_audio_in_video=use_audio_in_video)
            inputs = processor(
                text=text,
                audio=audios,
                images=images,
                videos=videos,
                return_tensors="pt",
                padding=True,
                use_audio_in_video=use_audio_in_video,
            )

            for key, value in list(inputs.items()):
                if not isinstance(value, torch.Tensor):
                    continue
                if key == "input_ids":
                    inputs[key] = value.to(device=execution_device, dtype=torch.int64)
                elif torch.is_floating_point(value):
                    inputs[key] = value.to(device=execution_device, dtype=model_dtype)
                else:
                    inputs[key] = value.to(device=execution_device)

            with torch.inference_mode():
                generated = model.generate(
                    **inputs,
                    return_audio=False,
                    use_audio_in_video=use_audio_in_video,
                    thinker_return_dict_in_generate=True,
                    thinker_max_new_tokens=max_new_tokens,
                    thinker_do_sample=do_sample,
                    thinker_temperature=max(temperature, 1e-5),
                    thinker_top_p=top_p,
                    thinker_top_k=top_k,
                    repetition_penalty=1.0,
                )

            generated_ids = generated[0] if isinstance(generated, tuple) else generated
            sequences = generated_ids.sequences if hasattr(generated_ids, "sequences") else generated_ids
            new_tokens = sequences[:, inputs["input_ids"].shape[1]:]
            return processor.batch_decode(
                new_tokens,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0].strip()
        finally:
            if inputs is not None:
                for key in tuple(inputs.keys()):
                    value = inputs.pop(key)
                    del value
                del inputs
            del new_tokens
            del sequences
            del generated_ids
            del generated
            del videos
            del images
            del audios
            del text
            self.clear_cached_memory()


def create_app(model_config: dict) -> FastAPI:
    app = FastAPI(title=f"tri-agent-{model_config['model']['key']}")
    app.state.model_config = model_config
    app.state.generate_lock = Lock()
    app.state.runner = Qwen3OmniRunner(model_config)

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "model_key": app.state.model_config["model"]["key"],
            "model_family": app.state.model_config["model"]["family"],
            "model_path": app.state.model_config["model"]["path"],
        }

    @app.post("/generate")
    def generate(request: GenerateRequest) -> dict[str, Any]:
        if not request.messages:
            raise HTTPException(status_code=400, detail="messages must not be empty")
        started = time.time()
        last_error: Exception | None = None
        for attempt in range(2):
            try:
                with app.state.generate_lock:
                    text = app.state.runner.generate_text(request.messages, request.generation)
                break
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                if attempt == 0 and app.state.runner._is_oom_error(exc):
                    print("[serve] generate retry after cuda oom", flush=True)
                    app.state.runner.clear_cached_memory()
                    continue
                traceback.print_exc()
                raise HTTPException(status_code=500, detail=f"Inference failed: {exc}") from exc
        else:
            raise HTTPException(status_code=500, detail=f"Inference failed: {last_error}")
        return {
            "text": text,
            "model_key": app.state.model_config["model"]["key"],
            "elapsed_sec": round(time.time() - started, 4),
        }

    return app


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Serve local Qwen3-Omni via a lightweight HTTP API.")
    parser.add_argument("--model-config", type=str, required=True)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    model_config = load_yaml_config(args.model_config)
    app = create_app(model_config)
    server_cfg = model_config.get("server", {})
    uvicorn.run(
        app,
        host=str(server_cfg.get("host", "0.0.0.0")),
        port=int(server_cfg.get("port", 18080)),
        log_level="info",
    )


if __name__ == "__main__":
    main()
