#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import torch
import torch.nn.functional as F
import numpy as np
import coremltools as ct
from transformers import AutoTokenizer
from mmengine.config import Config
from mmengine.runner import Runner
from mmdet.registry import MODELS

# —— Global Configuration —— #
os.environ["CUDA_VISIBLE_DEVICES"] = ""              # Force CPU usage
device = torch.device("cpu")

# Please adjust the following paths according to actual situation
CFG_PATH   = "configs/pretrain_v1/yolo_world_x_dual_vlpan_l2norm_2e-3_100e_4x8gpus_obj365v1_goldg_train_lvis_minival.py"
CKPT_PATH  = "pretrained_weights/yolo_world_x_clip_base_dual_vlpan_2e-3adamw_32xb16_100e_o365_goldg_cc3mlite_train_pretrained-8cf6b025.pth"
# Export filenames
TEXT_ENCODER_COREML     = "text_encoder.mlpackage"
VISUAL_DETECTOR_COREML  = "visual_detector.mlpackage"

# Text example (number of categories fixed to 1)
TEST_TEXT = "bottle"

# —— Step 0: Load and build YOLO-World PyTorch model —— #
cfg = Config.fromfile(CFG_PATH)
cfg.work_dir = "."
cfg.load_from = CKPT_PATH

runner = Runner.from_cfg(cfg)
runner.call_hook("before_run")
runner.load_or_resume()

model = MODELS.build(cfg.model)
checkpoint = torch.load(CKPT_PATH, map_location="cpu")
model.load_state_dict(checkpoint["state_dict"])
model.eval()


# —— Step 1: Define TextEncoderWrapper —— #
class TextEncoderWrapper(torch.nn.Module):
    """
    Extract CLIP text encoder from YOLO-World:
      Input:
        - input_ids      : Tensor[1, 77] (int64)
        - attention_mask : Tensor[1, 77] (int64)
      Output:
        - text_feats     : Tensor[1, 1, 512] (float32, L2 normalized)
    """
    def __init__(self, full_model):
        super().__init__()
        self.text_encoder = full_model.backbone.text_model.model.eval()

    def forward(self, input_ids, attention_mask):
        outputs = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        text_feat = outputs[0]                   # [1, 512]
        text_feat = text_feat.unsqueeze(1)       # 变成 [1, 1, 512]
        text_feat = F.normalize(text_feat, p=2, dim=-1)
        return text_feat                         # [1, 1, 512]


# —— Step 2: Define VisualDetectorWrapper —— #
class VisualDetectorWrapper(torch.nn.Module):
    """
    Extract visual detector from YOLO-World:
      Input:
        - image      : Tensor[1, 3, 640, 640] (float32)
        - text_feats : Tensor[1, 1, 512]       (float32)
        - txt_masks  : Tensor[1, 1]            (float32)
      Output:
        6 tensors (float32):
          - cls_score_0 : [1, 1, 80, 80]
          - cls_score_1 : [1, 1, 40, 40]
          - cls_score_2 : [1, 1, 20, 20]
          - bbox_pred_0 : [1, A0*4, 80, 80]
          - bbox_pred_1 : [1, A1*4, 40, 40]
          - bbox_pred_2 : [1, A2*4, 20, 20]
    """
    def __init__(self, full_model):
        super().__init__()
        self.image_model = full_model.backbone.image_model
        self.neck        = full_model.neck
        self.head_module = full_model.bbox_head.head_module

        # Hardcode pooling layers to ensure consistent dimensions during ONNX/CoreML inference
        pools = self.neck.text_enhancer.image_pools
        pools[0] = torch.nn.MaxPool2d(kernel_size=27, stride=27, padding=1)
        pools[1] = torch.nn.MaxPool2d(kernel_size=13, stride=13, padding=1)
        pools[2] = torch.nn.MaxPool2d(kernel_size=7,  stride=7,  padding=1)

    def forward(self, image, text_feats, txt_masks):
        img_feats   = self.image_model(image)             # [(1,C1,80,80), (1,C2,40,40), (1,C3,20,20)]
        fused_feats = self.neck(img_feats, text_feats)    # List: 3 scales of fused features
        cls_scores, bbox_preds = self.head_module(fused_feats, text_feats, txt_masks)
        return (
            cls_scores[0], cls_scores[1], cls_scores[2],
            bbox_preds[0], bbox_preds[1], bbox_preds[2]
        )


# —— Step 3: Convert TextEncoderWrapper to CoreML —— #
def convert_text_encoder_to_coreml():
    """
    Use PyTorch Wrapper to convert TextEncoderWrapper to CoreML (.mlpackage).
    Input: input_ids[1,77] (int64), attention_mask[1,77] (int64)
    Output: text_feats[1,1,512] (float32)
    """
    print("→ Converting TextEncoder (PyTorch) to CoreML...")

    wrapper_te = TextEncoderWrapper(model).eval()
    # 构造示例输入（只为 tracing，实际不会用到数据）
    tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32")
    encoding = tokenizer(
        [TEST_TEXT],
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=77
    )
    input_ids = encoding["input_ids"].to(device)          # Tensor[1,77], int64
    attention_mask = encoding["attention_mask"].to(device) # Tensor[1,77], int64

    # 先转换为 TorchScript 格式
    traced_model = torch.jit.trace(wrapper_te, (input_ids, attention_mask))

    # 使用 coremltools.convert 进行转换
    mlmodel = ct.convert(
        traced_model,
        inputs=[
            ct.TensorType(name="input_ids",      shape=input_ids.shape,      dtype=np.int64),
            ct.TensorType(name="attention_mask", shape=attention_mask.shape, dtype=np.int64)
        ],
        source="pytorch",
        minimum_deployment_target=ct.target.iOS15,
        compute_units=ct.ComputeUnit.CPU_AND_NE  # 指定计算单元为 CPU 和 Neural Engine
    )
    mlmodel.save(TEXT_ENCODER_COREML)
    print(f"[OK] TextEncoder CoreML model saved as '{TEXT_ENCODER_COREML}'")


# —— Step 4：将 VisualDetectorWrapper 转换为 CoreML —— #
def convert_visual_detector_to_coreml():
    """
    使用 PyTorch Wrapper，将 VisualDetectorWrapper 转为 CoreML (.mlmodel)。
    输入：image[1,3,640,640] (float32), text_feats[1,1,512] (float32), txt_masks[1,1] (float32)
    输出：6 路张量，与 PyTorch 输出保持一致。
    """
    print("→ Converting VisualDetector (PyTorch) to CoreML...")

    wrapper_vd = VisualDetectorWrapper(model).eval()

    # 构造 dummy 输入以确定形状（不用于实际推理）
    dummy_image      = torch.randn((1, 3, 640, 640), dtype=torch.float32)
    dummy_text_feats = torch.randn((1, 1, 512),       dtype=torch.float32)
    dummy_txt_masks  = torch.ones((1, 1),             dtype=torch.float32)

    # 先转换为 TorchScript 格式
    traced_model = torch.jit.trace(wrapper_vd, (dummy_image, dummy_text_feats, dummy_txt_masks))

    mlmodel = ct.convert(
        traced_model,
        inputs=[
            ct.TensorType(name="image",      shape=dummy_image.shape,      dtype=np.float32),
            ct.TensorType(name="text_feats", shape=dummy_text_feats.shape, dtype=np.float32),
            ct.TensorType(name="txt_masks",  shape=dummy_txt_masks.shape,  dtype=np.float32)
        ],
        source="pytorch",
        minimum_deployment_target=ct.target.iOS15,
        compute_units=ct.ComputeUnit.CPU_AND_NE  # 指定计算单元为 CPU 和 Neural Engine
    )
    mlmodel.save(VISUAL_DETECTOR_COREML)
    print(f"[OK] VisualDetector CoreML model saved as '{VISUAL_DETECTOR_COREML}'")


# —— 主程序 —— #
if __name__ == "__main__":
    convert_text_encoder_to_coreml()
    convert_visual_detector_to_coreml()
    print("\n[SUCCESS] PyTorch → CoreML conversion finished.")
