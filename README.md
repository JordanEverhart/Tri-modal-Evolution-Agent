# Tri-modal-Evolution-Agent

一个从零整理的、纯净的 `Qwen3-Omni-30B-A3B-Instruct` 推理工程。

当前目标：

- 统一跑 `WorldSense`、`Daily-Omni`、`OmniBench`
- 将模型路径、数据集路径、提示词、推理参数全部外置到 YAML
- 给未来新增数据集预留稳定扩展入口
- 提供本地 HTTP 推理服务和统一评测 CLI

## 目录结构

```text
Tri-modal-Evolution-Agent/
├── configs/
│   ├── datasets/
│   ├── models/
│   └── suites/
├── outputs/
├── src/tri_modal_evolution_agent/
│   ├── cli/
│   ├── config/
│   ├── datasets/
│   ├── engine/
│   └── evaluation/
└── tests/
```

## 快速开始

### 1. 安装

推荐在已有的 `qwen3omni` 环境内执行。当前机器的系统 `python` 是 `3.6.8`，太旧，不能直接跑本仓库。

建议显式使用：

- `/public/home/202492301216/Workplace/miniconda3/envs/qwen3omni/bin/python`
- `/public/home/202492301216/Workplace/miniconda3/envs/qwen3omni/bin/pip`

安装示例：

```bash
cd /public/home/202492301216/Workplace/Tri-modal-Evolution-Agent
/public/home/202492301216/Workplace/miniconda3/envs/qwen3omni/bin/pip install -e .
```

### 2. 启动本地 Qwen3Omni 服务

```bash
tri-agent-serve \
  --model-config configs/models/qwen3_omni_30b_a3b_instruct.yaml
```

也可以走统一入口：

```bash
tri-agent serve \
  --model-config configs/models/qwen3_omni_30b_a3b_instruct.yaml
```

默认会启动一个本地 HTTP 服务：

- `GET /health`
- `POST /generate`

### 3. 跑单个数据集

```bash
tri-agent-eval \
  --model-config configs/models/qwen3_omni_30b_a3b_instruct.yaml \
  --dataset-config configs/datasets/worldsense.yaml
```

也可以跑：

- `configs/datasets/daily_omni.yaml`
- `configs/datasets/omnibench.yaml`

统一入口写法：

```bash
tri-agent eval \
  --model-config configs/models/qwen3_omni_30b_a3b_instruct.yaml \
  --dataset-config configs/datasets/worldsense.yaml
```

### 4. 一次跑完整套 benchmark

```bash
tri-agent-run-suite \
  --model-config configs/models/qwen3_omni_30b_a3b_instruct.yaml \
  --suite-config configs/suites/default_benchmarks.yaml
```

统一入口写法：

```bash
tri-agent run-suite \
  --model-config configs/models/qwen3_omni_30b_a3b_instruct.yaml \
  --suite-config configs/suites/default_benchmarks.yaml
```

## 配置说明

### 模型 YAML

`configs/models/qwen3_omni_30b_a3b_instruct.yaml` 负责描述：

- 模型名和 family
- 本地模型路径
- 服务 host/port
- 默认生成参数
- 是否启用 `flash_attention_2`
- 是否关闭 talker

### 数据集 YAML

每个数据集 YAML 负责描述：

- loader 类型
- 数据集根目录与关键文件路径
- 系统提示词
- 用户提示模板
- 分组统计字段
- 该数据集默认推理参数

## 新增数据集

未来接新数据集时，建议按下面两步做：

1. 在 [`src/tri_modal_evolution_agent/datasets/registry.py`](/public/home/202492301216/Workplace/Tri-modal-Evolution-Agent/src/tri_modal_evolution_agent/datasets/registry.py) 注册一个新的 loader
2. 新建一份 `configs/datasets/<new_dataset>.yaml`

如果新数据集格式和现有某个 loader 足够接近，也可以直接复用已有 loader，只改 YAML。

## 输出

默认输出到：

```text
outputs/<dataset_key>/<run_name>/
```

其中会生成：

- `results.jsonl`
- `summary.json`
- `errors.jsonl`

## GitHub

仓库目标远端：

- `git@github.com:JordanEverhart/Tri-modal-Evolution-Agent.git`
- `https://github.com/JordanEverhart/Tri-modal-Evolution-Agent.git`

如果当前机器已经配置好 GitHub SSH 权限，我会直接完成初始化并推送。
