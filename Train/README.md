# Qwen3-Omni ms-swift Training

这个目录只放训练接入层，不放大数据、不放模型权重、不放 checkpoint。它面向 `ms-swift`，用于把本地 `Qwen3-Omni-30B-A3B-Instruct` 做成三段式训练接口：

- Stage 1: LoRA SFT
- Stage 2A: GRPO
- Stage 2B: GSPO

## 官方约束

本框架按 `ms-swift` 官方说明实现：

- 自定义数据集推荐直接传 `--dataset <dataset_path>`；标准格式使用 `messages`，多模态资源使用 `images`、`videos`、`audios` 字段。
- PPO/GRPO 数据只需要模型输入；奖励函数需要的额外字段，例如 `solution`，会原样传给 ORM。
- `Qwen/Qwen3-Omni-30B-A3B-Instruct` 的 `model_type` 是 `qwen3_omni_moe`。
- `qwen3_omni` 建议设置 `ENABLE_AUDIO_OUTPUT=False`；官方说明中这会只创建并微调 `thinker`，降低显存。
- 多模态 LoRA 中 `freeze_vit=true`、`freeze_aligner=true` 会避免给视觉/音频塔和 projector 添加 LoRA；`target_modules=all-linear` 默认作用在 LLM 侧。
- GSPO 在 `ms-swift` 中仍走 `swift rlhf --rlhf_type grpo`，通过 `--importance_sampling_level sequence` 切换。

参考：

- `ms-swift` repo: https://github.com/modelscope/ms-swift
- Custom dataset: https://github.com/modelscope/ms-swift/blob/main/docs/source_en/Customization/Custom-dataset.md
- RLHF/GRPO: https://github.com/modelscope/ms-swift/blob/main/docs/source_en/Instruction/RLHF.md
- Qwen3-Omni supported model: https://github.com/modelscope/ms-swift/blob/main/docs/source_en/Instruction/Supported-models-and-datasets.md
- GSPO: https://github.com/modelscope/ms-swift/blob/main/docs/source_en/Instruction/GRPO/AdvancedResearch/GSPO.md

## 目录

```text
Train/
├── configs/
│   ├── datasets/datasets.yaml
│   ├── env/default.yaml
│   ├── models/qwen3_omni_30b_a3b_instruct.yaml
│   └── recipes/
│       ├── lora_sft.yaml
│       ├── grpo.yaml
│       └── gspo.yaml
├── examples/jsonl/
├── schemas/
├── scripts/
└── src/qwen3omni_train/
```

## 快速使用

先安装这个轻量包：

```bash
cd /public/home/202492301216/Workplace/Tri-modal-Evolution-Agent/Train
/public/home/202492301216/Workplace/miniconda3/envs/qwen3omni/bin/pip install -e .
```

修改本地模型路径：

```text
configs/models/qwen3_omni_30b_a3b_instruct.yaml
```

修改训练数据路径：

```text
configs/datasets/datasets.yaml
```

只打印命令，不启动训练：

```bash
q3o-train launch --recipe configs/recipes/lora_sft.yaml --dry-run
q3o-train launch --recipe configs/recipes/grpo.yaml --dry-run
q3o-train launch --recipe configs/recipes/gspo.yaml --dry-run
```

真正执行训练需要在 GPU 计算节点上显式加 `--execute`：

```bash
q3o-train launch --recipe configs/recipes/lora_sft.yaml --execute
```

当前环境是 SSH 登录节点时，不要直接跑训练。先申请 GPU 资源，例如：

```bash
salloc -p gpuA800 --gres=gpu:1 --cpus-per-task=8 -w <主机号>
```

## 数据转换

输入 manifest 可以是 JSONL。字段足够简单时，converter 会转成 `ms-swift` 标准格式。
真实 manifest 默认不进 Git；仓库只保留 `manifests/*.example.jsonl`。
首次试跑 converter 可以先复制示例：

```bash
mkdir -p manifests
cp manifests/sft_manifest.example.jsonl manifests/sft_manifest.jsonl
cp manifests/grpo_manifest.example.jsonl manifests/grpo_manifest.jsonl
cp manifests/gspo_manifest.example.jsonl manifests/gspo_manifest.jsonl
```

SFT：

```bash
q3o-train convert --datasets configs/datasets/datasets.yaml --job sft
```

GRPO：

```bash
q3o-train convert --datasets configs/datasets/datasets.yaml --job grpo
```

GSPO：

```bash
q3o-train convert --datasets configs/datasets/datasets.yaml --job gspo
```

输出格式模板见：

- `schemas/ms_swift_sft.schema.json`
- `schemas/ms_swift_grpo.schema.json`
- `schemas/ms_swift_gspo.schema.json`
- `examples/jsonl/*.jsonl`

## 三个训练入口

这三个脚本只是薄包装，真实参数仍来自 YAML：

```bash
python scripts/train_lora_sft.py --dry-run
python scripts/train_grpo.py --dry-run
python scripts/train_gspo.py --dry-run
```

## 核心原则

- Git 仓库只存 `loader`、`converter`、`recipe`、`manifest`、`reward`、训练入口。
- 大数据只通过 YAML 指向外部路径。
- 模型权重只通过 YAML 指向外部路径。
- 输出默认写到 `Train/outputs/`，该目录应保持未纳入版本控制。
- LoRA SFT/GRPO/GSPO 都只训练 adapter；原始基座权重不被改写。
