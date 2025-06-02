#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer
import onnx
from onnx_tf.backend import prepare
import tensorflow as tf
import onnxruntime as ort
from PIL import Image
import numpy as np

# Disable GPU to avoid memory overflow during conversion/loading
os.environ["CUDA_VISIBLE_DEVICES"] = ""

from mmengine.config import Config
from mmengine.dataset import Compose
from mmengine.runner import Runner
from mmdet.registry import MODELS

cfg = Config.fromfile(
    "../../configs/yolo_world_x_dual_vlpan_l2norm_2e-3_100e_4x8gpus_obj365v1_goldg_train_lvis_minival.py"
)
cfg.work_dir = "."
cfg.load_from = "../../pretrained_models/yolo_world_x_clip_base_dual_vlpan_2e-3adamw_32xb16_100e_o365_goldg_cc3mlite_train_pretrained-8cf6b025.pth"
runner = Runner.from_cfg(cfg)
runner.call_hook("before_run")
runner.load_or_resume()
pipeline = cfg.test_dataloader.dataset.pipeline
runner.pipeline = Compose(pipeline)
runner.model.eval()

model = MODELS.build(cfg.model)
checkpoint = torch.load('./pretrained_models/yolo_world_x_clip_base_dual_vlpan_2e-3adamw_32xb16_100e_o365_goldg_cc3mlite_train_pretrained-8cf6b025.pth', map_location='cpu')
model.load_state_dict(checkpoint['state_dict'])

# -----------------------------------------------------------------------------
# Global filename configuration
# -----------------------------------------------------------------------------
TEXT_ENCODER_ONNX       = "text_encoder.onnx"
VISUAL_DETECTOR_ONNX    = "visual_detector.onnx"
TEXT_ENCODER_SAVED_DIR  = "text_encoder_saved_model"
VISUAL_DETECTOR_SAVED_DIR = "visual_detector_saved_model"
TEXT_ENCODER_TFLITE     = "text_encoder.tflite"
VISUAL_DETECTOR_TFLITE  = "visual_detector.tflite"

# Test image and text
IMAGE_PATH  = "./sample_images/bottles.png"          # Test image path
TEST_TEXT   = "bottle"                              # Fixed single text class

# -----------------------------------------------------------------------------
# Define PyTorch Wrapper: Text Encoder (fixed N=1)
# -----------------------------------------------------------------------------
class TextEncoderWrapper(nn.Module):
    """
    CLIP text encoder only, with fixed output shape [1,1,512]
    Input: input_ids ([1, seq_len]), attention_mask ([1, seq_len])
    Output: text_feats ([1, 1, 512]), L2 normalized
    """
    def __init__(self, model):
        super().__init__()
        self.text_encoder = model.backbone.text_model.model.eval()

    def forward(self, input_ids, attention_mask):
        outputs = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        text_feat = outputs[0]               # [1, 512]
        text_feat = F.normalize(text_feat, p=2, dim=-1)
        text_feat = text_feat.unsqueeze(0)   # [1, 1, 512]
        return text_feat                     # return 3D tensor

# -----------------------------------------------------------------------------
# Define PyTorch Wrapper: Visual Detector (fixed batch=1, N=1, and hardcoded Pooling parameters)
# -----------------------------------------------------------------------------
class VisualDetectorWrapper(nn.Module):
    """
    Visual detector component, inputs:
      image      : [1, 3, 640, 640]
      text_feats : [1, 1, 512] 
      txt_masks  : [1, 1]
    Outputs:
      cls_score_i: [1, 1, Hi, Wi]
      bbox_pred_i: [1, Ai*4, Hi, Wi]
    Hardcoded Pooling parameters for 80×80, 40×40, 20×20 in __init__:
      - 80→27: kernel=27,pad=1,stride=27
      - 40→13: kernel=13,pad=1,stride=13
      - 20→7:  kernel=7, pad=1,stride=7
    """
    def __init__(self, model):
        super().__init__()
        self.image_model = model.backbone.image_model
        self.neck = model.neck
        self.head_module = model.bbox_head.head_module

        # Fix three Pooling layers as constants
        pools = self.neck.text_enhancer.image_pools
        pools[0] = nn.MaxPool2d(kernel_size=(27, 27), stride=(27, 27), padding=(1, 1))
        pools[1] = nn.MaxPool2d(kernel_size=(13, 13), stride=(13, 13), padding=(1, 1))
        pools[2] = nn.MaxPool2d(kernel_size=(7, 7),   stride=(7, 7),   padding=(1, 1))

    def forward(self, image, text_feats, txt_masks):
        """
        image      : [1, 3, 640, 640]
        text_feats : [1, 1, 512]
        txt_masks  : [1, 1]
        """
        img_feats = self.image_model(image)           # [1,C1,80,80], [1,C2,40,40], [1,C3,20,20]
        fused_feats = self.neck(img_feats, text_feats)
        cls_scores, bbox_preds = self.head_module(fused_feats, text_feats, txt_masks)
        return (
            cls_scores[0], cls_scores[1], cls_scores[2],
            bbox_preds[0], bbox_preds[1], bbox_preds[2]
        )

# -----------------------------------------------------------------------------
# Step 1 & 2: Export text_encoder.onnx and convert to TFLite
# -----------------------------------------------------------------------------
def export_text_encoder_to_onnx_and_tflite():
    """
    1. Export text_encoder.onnx (input: [1,77], output: [1,1,512])
    2. Convert ONNX to SavedModel, then to text_encoder.tflite
    """
    print("\n=== STEP 1: Exporting text_encoder.onnx ===")
    wrapper_te = TextEncoderWrapper(model).eval()

    tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32")
    encoding = tokenizer(
        [TEST_TEXT],            # Single text only
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=77
    )
    input_ids = encoding["input_ids"]             # [1, 77]
    attention_mask = encoding["attention_mask"]   # [1, 77]

    with torch.no_grad():
        torch.onnx.export(
            wrapper_te,
            (input_ids, attention_mask),
            TEXT_ENCODER_ONNX,
            input_names=["input_ids", "attention_mask"],
            output_names=["text_feats"],
            opset_version=11,
            do_constant_folding=True
        )
    print(f"[DONE] Generated {TEXT_ENCODER_ONNX}")

    print("\n=== STEP 2: Converting ONNX → SavedModel → TFLite (text_encoder) ===")
    onnx_model = onnx.load(TEXT_ENCODER_ONNX)
    tf_rep = prepare(onnx_model)
    if os.path.isdir(TEXT_ENCODER_SAVED_DIR):
        tf.io.gfile.rmtree(TEXT_ENCODER_SAVED_DIR)
    tf_rep.export_graph(TEXT_ENCODER_SAVED_DIR)
    print(f"[INFO] SavedModel saved to: {TEXT_ENCODER_SAVED_DIR}")

    converter_te = tf.lite.TFLiteConverter.from_saved_model(TEXT_ENCODER_SAVED_DIR)
    converter_te.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
    tflite_te = converter_te.convert()
    with open(TEXT_ENCODER_TFLITE, "wb") as f:
        f.write(tflite_te)
    print(f"[DONE] Generated {TEXT_ENCODER_TFLITE}")

# -----------------------------------------------------------------------------
# Step 3 & 4: Export visual_detector.onnx and convert to TFLite  
# -----------------------------------------------------------------------------
def export_visual_detector_to_onnx_and_tflite():
    """
    1. Export visual_detector.onnx (input: [1,3,640,640], [1,1,512], [1,1])
       With fixed pooling parameters to ensure correct concat dimensions in TFLite
    2. Convert ONNX to SavedModel, then to visual_detector.tflite
    """
    print("\n=== STEP 3: Exporting visual_detector.onnx ===")
    wrapper_vd = VisualDetectorWrapper(model).eval()

    dummy_image = torch.randn(1, 3, 640, 640)    # [1,3,640,640]
    dummy_text_feats = torch.randn(1, 1, 512)    # [1,1,512]
    dummy_txt_masks = torch.ones(1, 1, dtype=torch.float32)  # [1,1]

    with torch.no_grad():
        torch.onnx.export(
            wrapper_vd,
            (dummy_image, dummy_text_feats, dummy_txt_masks),
            VISUAL_DETECTOR_ONNX,
            input_names=["image", "text_feats", "txt_masks"],
            output_names=[
                "cls_score_0", "cls_score_1", "cls_score_2",
                "bbox_pred_0", "bbox_pred_1", "bbox_pred_2"
            ],
            opset_version=12,
            do_constant_folding=True
        )
    print(f"[DONE] Generated {VISUAL_DETECTOR_ONNX}")

    print("\n=== STEP 4: Converting ONNX → SavedModel → TFLite (visual_detector) ===")
    onnx_model = onnx.load(VISUAL_DETECTOR_ONNX)
    tf_rep = prepare(onnx_model)
    if os.path.isdir(VISUAL_DETECTOR_SAVED_DIR):
        tf.io.gfile.rmtree(VISUAL_DETECTOR_SAVED_DIR)
    tf_rep.export_graph(VISUAL_DETECTOR_SAVED_DIR)
    print(f"[INFO] SavedModel saved to: {VISUAL_DETECTOR_SAVED_DIR}")

    converter_vd = tf.lite.TFLiteConverter.from_saved_model(VISUAL_DETECTOR_SAVED_DIR)
    converter_vd.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
    tflite_vd = converter_vd.convert()
    with open(VISUAL_DETECTOR_TFLITE, "wb") as f:
        f.write(tflite_vd)
    print(f"[DONE] Generated {VISUAL_DETECTOR_TFLITE}")

# -----------------------------------------------------------------------------
# Step 5: Run inference test with TFLite models (all fixed size)
# -----------------------------------------------------------------------------
def run_tflite_inference():
    """Run TFLite inference - moved to TFLite_inference.py"""
    pass

# -----------------------------------------------------------------------------
# Main program entry
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    export_text_encoder_to_onnx_and_tflite()
    export_visual_detector_to_onnx_and_tflite()
    print("\n[INFO] Model export complete, use TFLite_inference.py for inference testing\n")
