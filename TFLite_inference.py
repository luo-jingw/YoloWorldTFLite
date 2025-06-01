#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
optimized_inference_with_reference_postprocess.py

Optimized TFLite inference script with integrated reference post-processing:
1. Text encoding (TFLite)
2. Visual detection (TFLite)
3. PyTorch reference implementation: multi-scale decoding + normalized threshold + NMS + visualization

Dependencies:
  pip install numpy tensorflow Pillow transformers torch torchvision opencv-python matplotlib
"""

import os
import sys
from typing import Tuple, List

import numpy as np
import tensorflow as tf
import torch
from torchvision.ops import nms
from PIL import Image
from transformers import AutoTokenizer
import cv2
import matplotlib.pyplot as plt

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
TEXT_ENCODER_TFLITE     = "text_encoder.tflite"
VISUAL_DETECTOR_TFLITE  = "visual_detector.tflite"
IMAGE_PATH              = "./sample_images/bottles.png"
TEST_TEXT               = "bottle"

MIN_THRESH              = 0.05    # Original score max threshold
NORM_THRESH            = 0.5     # Normalized threshold
NMS_IOU_THRESH         = 0.5     # NMS IOU threshold

TARGET_SIZE            = (640, 640)  # Fixed input size

# Single class label list
CLASS_NAMES = ["bottle"]

# -----------------------------------------------------------------------------
# Global resource initialization
# -----------------------------------------------------------------------------
interpreter_te = None
interpreter_vd = None
tokenizer = AutoTokenizer.from_pretrained("./tokenizer")  # 修改为本地tokenizer路径


def init_interpreters():
    """Load and cache two TFLite Interpreters."""
    global interpreter_te, interpreter_vd

    if interpreter_te is None:
        if not os.path.isfile(TEXT_ENCODER_TFLITE):
            print(f"[ERROR] Cannot find {TEXT_ENCODER_TFLITE}")
            sys.exit(1)
        interpreter_te = tf.lite.Interpreter(model_path=TEXT_ENCODER_TFLITE)
        interpreter_te.allocate_tensors()

    if interpreter_vd is None:
        if not os.path.isfile(VISUAL_DETECTOR_TFLITE):
            print(f"[ERROR] Cannot find {VISUAL_DETECTOR_TFLITE}")
            sys.exit(1)
        interpreter_vd = tf.lite.Interpreter(model_path=VISUAL_DETECTOR_TFLITE)
        # Delayed allocation, call allocate_tensors() after resize


# -----------------------------------------------------------------------------
# 图像与文本预处理
# -----------------------------------------------------------------------------
def load_image(image_path: str, target_size: Tuple[int, int] = TARGET_SIZE) -> Tuple[np.ndarray, np.ndarray]:
    """
    Read image, resize to target_size, normalize to [0,1],
    convert to NCHW numpy array format for TFLite.
    Returns: numpy.ndarray([1,3,640,640]), original RGB array for visualization
    """
    img_pil = Image.open(image_path).convert("RGB")
    img_resized = img_pil.resize(target_size)
    img_np = np.array(img_resized, dtype=np.float32) / 255.0   # H×W×C, [0,1]
    img_nchw = np.transpose(img_np, (2, 0, 1))[None, ...]      # 1×C×H×W
    return img_nchw, np.array(img_pil)


def encode_text(text: str) -> np.ndarray:
    """
    Use TFLite text encoder to map single text to feature vector.
    Returns float32 numpy array with shape [1, 1, 512].
    """
    encoding = tokenizer(
        [text],
        return_tensors="np",
        padding="max_length",
        truncation=True,
        max_length=77
    )
    input_ids = encoding["input_ids"].astype(np.int64)           # [1, 77]
    attention_mask = encoding["attention_mask"].astype(np.int64) # [1, 77]

    input_details = interpreter_te.get_input_details()
    output_details = interpreter_te.get_output_details()
    idx_iids = next(d["index"] for d in input_details if "input_ids" in d["name"])
    idx_imask = next(d["index"] for d in input_details if "attention_mask" in d["name"])
    idx_out  = output_details[0]["index"]  # text_feats 输出

    interpreter_te.set_tensor(idx_iids, input_ids)
    interpreter_te.set_tensor(idx_imask, attention_mask)
    interpreter_te.invoke()

    text_feats = interpreter_te.get_tensor(idx_out)  # [1,1,512]
    return text_feats


def detect_image(image_np: np.ndarray, text_feats: np.ndarray) -> Tuple[np.ndarray, ...]:
    """
    Run inference using TFLite visual detector
    Returns: Tuple[cls_score_20x20, cls_score_40x40, cls_score_80x80,
                  bbox_pred_20x20, bbox_pred_40x40, bbox_pred_80x80]
    """
    input_details = interpreter_vd.get_input_details()
    output_details = interpreter_vd.get_output_details()

    txt_masks = np.ones((1, 1), dtype=np.float32)
    name_to_idx = {d["name"]: d["index"] for d in input_details}

    # Resize 动态输入
    interpreter_vd.resize_tensor_input(name_to_idx["serving_default_txt_masks:0"], txt_masks.shape)
    interpreter_vd.resize_tensor_input(name_to_idx["serving_default_text_feats:0"], text_feats.shape)
    interpreter_vd.resize_tensor_input(name_to_idx["serving_default_image:0"], image_np.shape)
    interpreter_vd.allocate_tensors()

    interpreter_vd.set_tensor(name_to_idx["serving_default_txt_masks:0"], txt_masks)
    interpreter_vd.set_tensor(name_to_idx["serving_default_text_feats:0"], text_feats.astype(np.float32))
    interpreter_vd.set_tensor(name_to_idx["serving_default_image:0"], image_np)
    interpreter_vd.invoke()

    # 直接返回排序好的输出元组
    outputs = []
    for detail in output_details:
        tensor = interpreter_vd.get_tensor(detail["index"])
        print(f"\n[DEBUG] Output shape: {tensor.shape}, min/max: {tensor.min():.4f}/{tensor.max():.4f}")
        outputs.append(tensor)
    
    # 按特征图大小排序: [20x20, 40x40, 80x80]
    cls_outputs = [outputs[5], outputs[4], outputs[2]]   # 分类输出
    bbox_outputs = [outputs[3], outputs[0], outputs[1]]  # 边界框输出
    return tuple(cls_outputs + bbox_outputs)


# -----------------------------------------------------------------------------
# 后处理：参考代码实现
# -----------------------------------------------------------------------------
def post_process_reference(outputs: Tuple[np.ndarray, ...],
                         image_rgb: np.ndarray,
                         min_thresh: float = MIN_THRESH,
                         norm_thresh: float = NORM_THRESH,
                         nms_thresh: float = NMS_IOU_THRESH
                         ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Reference post-processing implementation
    Args:
        outputs: (cls_20x20, cls_40x40, cls_80x80, bbox_20x20, bbox_40x40, bbox_80x80)
    """
    all_boxes = []
    all_cls_probs = []
    
    # 解包输出
    cls_outputs = outputs[:3]    # 20x20, 40x40, 80x80
    bbox_outputs = outputs[3:]   # 20x20, 40x40, 80x80
    
    for level, (cls_score_np, bbox_pred_np) in enumerate(zip(cls_outputs, bbox_outputs)):
        cls_score = torch.from_numpy(cls_score_np).float()
        bbox_pred = torch.from_numpy(bbox_pred_np).float()

        # 分类分数 shape: [1, C, H, W] → sigmoid → [C, H, W] → permute → [H, W, C] → reshape → [H*W, C]
        cls_prob = torch.sigmoid(cls_score)[0].permute(1, 2, 0).reshape(-1, cls_score.shape[1])
        all_cls_probs.append(cls_prob)  # list of [H*W, C]

        # 边框回归 shape: [1,4,H,W] → [4, H, W]
        bbox = bbox_pred[0]
        tx, ty, tr, tb = bbox[0], bbox[1], bbox[2], bbox[3]

        H, W = tx.shape
        stride = TARGET_SIZE[0] / H  # 640/H

        grid_y, grid_x = torch.meshgrid(torch.arange(H), torch.arange(W), indexing="ij")
        x_center = (grid_x + 0.5) * stride
        y_center = (grid_y + 0.5) * stride

        x1 = x_center - tx * stride
        y1 = y_center - ty * stride
        x2 = x_center + tr * stride
        y2 = y_center + tb * stride

        boxes = torch.stack([x1, y1, x2, y2], dim=-1).reshape(-1, 4)  # [H*W, 4]
        all_boxes.append(boxes)

    if not all_boxes:
        return torch.empty((0, 4)), torch.empty((0,)), torch.empty((0,), dtype=torch.long)

    all_boxes_tensor     = torch.cat(all_boxes, dim=0)     # [T, 4]
    all_cls_probs_tensor = torch.cat(all_cls_probs, dim=0) # [T, C]

    final_boxes = []
    final_scores = []
    final_labels = []

    num_classes = all_cls_probs_tensor.shape[1]
    for cls_id in range(num_classes):
        raw_scores = all_cls_probs_tensor[:, cls_id]
        if raw_scores.max() < min_thresh:
            continue

        min_s, max_s = raw_scores.min(), raw_scores.max()
        norm_scores = (raw_scores - min_s) / (max_s - min_s + 1e-6)
        mask = norm_scores > norm_thresh
        if mask.sum() == 0:
            continue

        boxes_cls = all_boxes_tensor[mask]
        scores_cls = raw_scores[mask]

        keep = nms(boxes_cls, scores_cls, iou_threshold=nms_thresh)
        final_boxes.append(boxes_cls[keep])
        final_scores.append(scores_cls[keep])
        final_labels.append(torch.full_like(scores_cls[keep], cls_id, dtype=torch.long))

    if final_boxes:
        final_boxes  = torch.cat(final_boxes, dim=0)
        final_scores = torch.cat(final_scores, dim=0)
        final_labels = torch.cat(final_labels, dim=0)
        print(f"[INFO] After NMS: {final_boxes.shape[0]} boxes kept")
    else:
        print("[WARN] No boxes passed filtering.")
        final_boxes  = torch.empty((0, 4))
        final_scores = torch.empty((0,))
        final_labels = torch.empty((0,), dtype=torch.long)

    # Scale back to original image
    H_orig, W_orig = image_rgb.shape[:2]
    scale_x = W_orig / TARGET_SIZE[0]
    scale_y = H_orig / TARGET_SIZE[1]
    scaled_boxes = final_boxes.clone()
    if scaled_boxes.numel() > 0:
        scaled_boxes[:, [0, 2]] *= scale_x
        scaled_boxes[:, [1, 3]] *= scale_y

    return scaled_boxes, final_scores, final_labels


# -----------------------------------------------------------------------------
# 可视化函数
# -----------------------------------------------------------------------------
def visualize(image_rgb: np.ndarray,
              boxes: torch.Tensor,
              scores: torch.Tensor,
              labels: torch.Tensor,
              class_names: List[str],
              save_path: str = "detection_result.jpg") -> None:
    """
    Draw detection results on original RGB image and save/display.
    """
    img = image_rgb.copy()
    H, W = img.shape[:2]
    font_scale = max(0.5, min(W, H) / 800 * 1.0)
    thickness = max(1, int(min(W, H) / 400))

    for box, score, cls_id in zip(boxes, scores, labels):
        x1, y1, x2, y2 = box.int().tolist()
        label_text = f"{class_names[cls_id]} {score:.2f}"

        # 绘制框
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), thickness)

        # 绘制标签背景
        (text_w, text_h), baseline = cv2.getTextSize(label_text,
                                                     cv2.FONT_HERSHEY_SIMPLEX,
                                                     font_scale, thickness)
        cv2.rectangle(img,
                      (x1, y1 - text_h - baseline - 4),
                      (x1 + text_w + 4, y1),
                      (0, 255, 0), -1)
        # 绘制文字
        cv2.putText(img, label_text, (x1 + 2, y1 - baseline - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0),
                    thickness, cv2.LINE_AA)

    cv2.imwrite(save_path, img)
    print(f"[INFO] Detection result saved as {save_path}")

    plt.figure(figsize=(10, 8))
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.title("Detection Result")
    plt.show()


# -----------------------------------------------------------------------------
# 主流程
# -----------------------------------------------------------------------------
def main():
    # Check model and image files
    if not os.path.isfile(TEXT_ENCODER_TFLITE):
        print(f"[ERROR] Cannot find {TEXT_ENCODER_TFLITE}")
        sys.exit(1)
    if not os.path.isfile(VISUAL_DETECTOR_TFLITE):
        print(f"[ERROR] Cannot find {VISUAL_DETECTOR_TFLITE}")
        sys.exit(1)
    if not os.path.isfile(IMAGE_PATH):
        print(f"[ERROR] Cannot find test image: {IMAGE_PATH}")
        sys.exit(1)

    # 初始化 Interpreters
    init_interpreters()

    # 文本编码
    print("\n=== 文本编码 ===")
    text_feats = encode_text(TEST_TEXT)  # [1,1,512]
    print(f"text_feats.shape = {text_feats.shape}")

    # 图像预处理
    print("\n=== 图像预处理 ===")
    image_np, image_rgb = load_image(IMAGE_PATH)  # image_rgb 用于可视化
    print(f"image_np.shape = {image_np.shape}")

    # 视觉检测
    print("\n=== 视觉检测 ===")
    outputs = detect_image(image_np, text_feats)
    print(f"\n检测到 {len(outputs)} 个特征图输出")
    for i, tensor in enumerate(outputs):
        print(f"  Output {i}: shape={tensor.shape}")
    
    # 后处理
    print("\n=== 后处理 ===")
    scaled_boxes, final_scores, final_labels = post_process_reference(outputs, image_rgb)
    if scaled_boxes.numel() == 0:
        print("[INFO] 未检测到任何目标。")
        return

    print(f"[INFO] 检测到 {scaled_boxes.shape[0]} 个目标")
    for i, score in enumerate(final_scores, start=1):
        print(f"  {i}. {TEST_TEXT} (conf={score:.2f})")

    # 可视化
    visualize(image_rgb, scaled_boxes, final_scores, final_labels, CLASS_NAMES)


if __name__ == "__main__":
    main()
    print("\n[ALL DONE]\n")
