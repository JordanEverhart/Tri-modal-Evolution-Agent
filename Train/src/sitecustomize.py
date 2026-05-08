"""Local compatibility patches loaded automatically by Python.

The Qwen3-Omni vLLM fork still imports ``VideoInput`` from
``transformers.image_utils`` through the Qwen2.5-Omni processor path, while
Transformers 4.57 exposes it from ``transformers.video_utils``. Keep the alias
local to this training project instead of editing the conda package.
"""

try:
    from transformers import image_utils, video_utils
except Exception:
    image_utils = None
    video_utils = None

if image_utils is not None and video_utils is not None and not hasattr(image_utils, "VideoInput"):
    image_utils.VideoInput = video_utils.VideoInput
