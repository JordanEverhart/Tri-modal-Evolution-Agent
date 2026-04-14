from __future__ import annotations

import argparse
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
        kwargs: dict[str, Any] = {
            "torch_dtype": "auto",
            "device_map": server_cfg.get("device_map", "auto"),
        }
        if server_cfg.get("flash_attn2", False):
            kwargs["attn_implementation"] = "flash_attention_2"

        model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(model_path, **kwargs)
        if server_cfg.get("disable_talker", False):
            model.disable_talker()
        model.eval()
        processor = Qwen3OmniMoeProcessor.from_pretrained(model_path)

        return {
            "torch": torch,
            "process_mm_info": process_mm_info,
            "model": model,
            "processor": processor,
        }

    def generate_text(self, messages: list[dict[str, Any]], generation: dict[str, Any]) -> str:
        torch = self.runner["torch"]
        process_mm_info = self.runner["process_mm_info"]
        model = self.runner["model"]
        processor = self.runner["processor"]

        use_audio_in_video = bool(generation.get("use_audio_in_video", True))
        max_new_tokens = int(generation.get("max_new_tokens", 64))
        do_sample = bool(generation.get("do_sample", False))
        temperature = float(generation.get("temperature", 0.0))
        top_p = float(generation.get("top_p", 1.0))
        top_k = int(generation.get("top_k", 20))

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

        device = next(model.parameters()).device
        dtype = next(model.parameters()).dtype
        for key, value in list(inputs.items()):
            if not isinstance(value, torch.Tensor):
                continue
            if key == "input_ids":
                inputs[key] = value.to(device=device, dtype=torch.int64)
            elif torch.is_floating_point(value):
                inputs[key] = value.to(device=device, dtype=dtype)
            else:
                inputs[key] = value.to(device=device)

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
        try:
            with app.state.generate_lock:
                text = app.state.runner.generate_text(request.messages, request.generation)
        except Exception as exc:  # noqa: BLE001
            traceback.print_exc()
            raise HTTPException(status_code=500, detail=f"Inference failed: {exc}") from exc
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
