"""Person re-identification embeddings for detected boxes (OSNet through boxmot).

Colour histograms tell the two teams apart but hardly the players within a team (same bibs). A ReID network
trained on many people's crops gives a 512-d embedding where the same person in two frames, or in the two
cameras, is close; the box-level tracker uses it to link tracklets and to pair the cameras (pipeline.align).

Install (Python 3.14; boxmot 12's own pins - numpy==1.26.4, opencv-python<5, torchvision<0.18 - do not install
there and would replace this project's numpy / OpenCV, so boxmot goes in without its dependencies):

    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128    # GPU build (or plain pip: CPU)
    pip install -e ".[reid]"            # the light deps boxmot's ReID path imports: filterpy ftfy gdown lap loguru
                                        # pandas pyyaml regex scikit-learn (+ torch / torchvision if not yet there)
    pip install --no-deps boxmot        # tested with 12.0.2

boxmot asks for lapx; the `lap` package provides the same module and installs on 3.14. Weights: FUTSAL_REID_WEIGHTS,
else C:/Users/user/dev/models/osnet_ain_x1_0_msmt17.pt, else boxmot downloads them on first use.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Callable

import numpy as np

DEFAULT_WEIGHTS = "osnet_ain_x1_0_msmt17.pt"
LOCAL_WEIGHTS = Path("C:/Users/user/dev/models") / DEFAULT_WEIGHTS
DIM = 512


def default_weights() -> str:
    """FUTSAL_REID_WEIGHTS, else the local model folder, else the bare name (boxmot downloads it)."""
    env = os.environ.get("FUTSAL_REID_WEIGHTS")
    if env:
        return env
    return str(LOCAL_WEIGHTS) if LOCAL_WEIGHTS.exists() else DEFAULT_WEIGHTS


def reid_embedder(weights: str | None = None, device: str | None = None,
                  half: bool = True) -> Callable[[np.ndarray, np.ndarray], np.ndarray]:
    """embed(frame_bgr, xyxy (n, 4) pixels) -> (n, 512) float32, L2-normalised. Loads the model once.

    `device` defaults to the GPU when there is one; half precision is used on the GPU only."""
    try:
        import torch
        from boxmot.appearance.reid_auto_backend import ReidAutoBackend
        from boxmot.utils import WEIGHTS
    except ImportError as e:
        raise ImportError("ReID 임베딩에는 torch와 boxmot이 필요합니다: "
                          'pip install -e ".[reid]" 후 pip install --no-deps boxmot '
                          "(자세한 설치 순서는 futsal/pipeline/reid.py 맨 위 설명)") from e
    w = Path(weights or default_weights())
    if w.parent == Path(".") and not w.exists():
        w = WEIGHTS / w.name                      # boxmot downloads known models into its own weights folder
    dev = torch.device(device or ("cuda:0" if torch.cuda.is_available() else "cpu"))
    model = ReidAutoBackend(weights=w, device=dev, half=half and dev.type == "cuda").model

    def embed(frame: np.ndarray, xyxy: np.ndarray) -> np.ndarray:
        boxes = np.asarray(xyxy, float).reshape(-1, 4)
        if not len(boxes):
            return np.zeros((0, DIM), np.float32)
        # boxmot crops with the box as given: keep every box inside the frame and at least 2 px wide and tall,
        # or an empty crop crashes the resize
        H, W = frame.shape[:2]
        b = boxes.copy()
        b[:, [0, 2]] = np.clip(b[:, [0, 2]], 0, W - 2)
        b[:, [1, 3]] = np.clip(b[:, [1, 3]], 0, H - 2)
        b[:, 2] = np.maximum(b[:, 2], b[:, 0] + 2)
        b[:, 3] = np.maximum(b[:, 3], b[:, 1] + 2)
        f = np.asarray(model.get_features(b, frame), np.float32).reshape(len(b), -1)
        return f / np.maximum(np.linalg.norm(f, axis=1, keepdims=True), 1e-9)
    return embed
