"""YOLOv6 CUDA detector for single-frame inference."""

from __future__ import annotations

import math
import sys
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
YOLOV6_ROOT = ROOT / "third_party" / "YOLOv6"
if str(YOLOV6_ROOT) not in sys.path:
    sys.path.insert(0, str(YOLOV6_ROOT))

# PyTorch >=2.6 defaults weights_only=True; YOLOv6 checkpoints need full unpickle.
_orig_torch_load = torch.load


def _torch_load_compat(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _orig_torch_load(*args, **kwargs)


torch.load = _torch_load_compat

from yolov6.data.data_augment import letterbox  # noqa: E402
from yolov6.layers.common import DetectBackend, RepVGGBlock  # noqa: E402
from yolov6.utils.events import load_yaml  # noqa: E402
from yolov6.utils.nms import non_max_suppression  # noqa: E402


class YOLOv6Detector:
    def __init__(
        self,
        weights: str | Path,
        device: str = "0",
        img_size: int = 640,
        conf_thres: float = 0.4,
        iou_thres: float = 0.45,
        half: bool = True,
        yaml_path: str | Path | None = None,
    ) -> None:
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.max_det = 100

        cuda = device != "cpu" and torch.cuda.is_available()
        self.device = torch.device(f"cuda:{device}" if cuda else "cpu")
        self.half = bool(half and self.device.type == "cuda")

        yaml_path = Path(yaml_path or YOLOV6_ROOT / "data" / "coco.yaml")
        self.class_names = load_yaml(str(yaml_path))["names"]

        self.model = DetectBackend(str(weights), device=self.device)
        self.stride = int(self.model.stride)
        self.img_size = self._check_img_size(img_size, self.stride)
        self._switch_to_deploy(self.model.model)

        if self.half:
            self.model.model.half()
        else:
            self.model.model.float()

        dummy = torch.zeros(1, 3, *self.img_size).to(self.device)
        dummy = dummy.half() if self.half else dummy.float()
        self.model(dummy)

        self._fps = deque(maxlen=30)
        self.last_fps = 0.0

    @staticmethod
    def _switch_to_deploy(model: torch.nn.Module) -> None:
        for layer in model.modules():
            if isinstance(layer, RepVGGBlock):
                layer.switch_to_deploy()
            elif isinstance(layer, torch.nn.Upsample) and not hasattr(
                layer, "recompute_scale_factor"
            ):
                layer.recompute_scale_factor = None

    @staticmethod
    def _make_divisible(x: int, divisor: int) -> int:
        return math.ceil(x / divisor) * divisor

    def _check_img_size(self, img_size: int | list[int], stride: int) -> list[int]:
        if isinstance(img_size, int):
            size = self._make_divisible(img_size, stride)
            return [size, size]
        return [self._make_divisible(x, stride) for x in img_size]

    @staticmethod
    def _rescale(ori_shape, boxes, target_shape):
        ratio = min(ori_shape[0] / target_shape[0], ori_shape[1] / target_shape[1])
        padding = (
            (ori_shape[1] - target_shape[1] * ratio) / 2,
            (ori_shape[0] - target_shape[0] * ratio) / 2,
        )
        boxes[:, [0, 2]] -= padding[0]
        boxes[:, [1, 3]] -= padding[1]
        boxes[:, :4] /= ratio
        boxes[:, 0].clamp_(0, target_shape[1])
        boxes[:, 1].clamp_(0, target_shape[0])
        boxes[:, 2].clamp_(0, target_shape[1])
        boxes[:, 3].clamp_(0, target_shape[0])
        return boxes

    @staticmethod
    def _color(i: int) -> tuple[int, int, int]:
        palette = (
            (56, 56, 255),
            (151, 157, 255),
            (31, 112, 255),
            (29, 178, 255),
            (49, 210, 207),
            (10, 249, 72),
            (23, 204, 146),
            (134, 219, 61),
            (52, 147, 26),
            (187, 212, 0),
            (168, 153, 44),
            (255, 194, 0),
            (147, 69, 52),
            (255, 115, 100),
            (236, 24, 0),
            (255, 56, 132),
            (133, 0, 82),
            (255, 56, 203),
            (200, 149, 255),
            (199, 55, 255),
        )
        return palette[int(i) % len(palette)]

    def infer(self, frame_bgr: np.ndarray) -> np.ndarray:
        img_src = frame_bgr
        image = letterbox(img_src, self.img_size, stride=self.stride)[0]
        image = image.transpose((2, 0, 1))[::-1]
        image = np.ascontiguousarray(image)
        tensor = torch.from_numpy(image).to(self.device)
        tensor = tensor.half() if self.half else tensor.float()
        tensor /= 255.0
        if tensor.ndimension() == 3:
            tensor = tensor.unsqueeze(0)

        t0 = time.time()
        with torch.inference_mode():
            pred = self.model(tensor)
            det = non_max_suppression(
                pred,
                self.conf_thres,
                self.iou_thres,
                classes=None,
                agnostic=False,
                max_det=self.max_det,
            )[0]
        elapsed = max(time.time() - t0, 1e-6)
        self._fps.append(1.0 / elapsed)
        self.last_fps = float(np.mean(self._fps))

        out = img_src.copy()
        if len(det):
            det = det.clone()
            det[:, :4] = self._rescale(tensor.shape[2:], det[:, :4], img_src.shape).round()
            for *xyxy, conf, cls_id in reversed(det):
                cls_i = int(cls_id)
                label = f"{self.class_names[cls_i]} {float(conf):.2f}"
                color = self._color(cls_i)
                p1 = (int(xyxy[0]), int(xyxy[1]))
                p2 = (int(xyxy[2]), int(xyxy[3]))
                cv2.rectangle(out, p1, p2, color, 2, cv2.LINE_AA)
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
                y0 = max(0, p1[1] - th - 6)
                cv2.rectangle(out, (p1[0], y0), (p1[0] + tw + 4, p1[1]), color, -1)
                cv2.putText(
                    out,
                    label,
                    (p1[0] + 2, p1[1] - 4),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )

        cv2.rectangle(out, (8, 8), (250, 42), (0, 0, 0), -1)
        cv2.putText(
            out,
            f"YOLOv6 CUDA | {self.last_fps:.1f} FPS",
            (14, 32),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (80, 255, 80),
            2,
            cv2.LINE_AA,
        )
        return out
