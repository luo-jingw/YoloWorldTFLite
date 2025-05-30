import numpy as np
import cv2 as cv
import tensorflow as tf
import torch
from torchvision.ops import nms
import matplotlib.pyplot as plt
from transformers import AutoTokenizer
import time

def yolo_world_detect(texts, image_path, output_path="output.jpg", 
                     score_threshold=0.05, nms_threshold=0.5, confidence_threshold=0.25):
    """
    YOLO-World Object Detection Function
    
    Args:
        texts (list): List of detection target texts
        image_path (str): Input image path
        output_path (str): Output image save path
        score_threshold (float): Threshold to filter out boxes with low confidence
        nms_threshold (float): IoU threshold for NMS
        confidence_threshold (float): Final confidence threshold
        
    Returns:
        dict: Dictionary containing detection results and time statistics
    """
    # time list
    time_stamps = np.zeros(8)
    time_stamps[0] = time.time()

    ### Load models

    # load tokenizer from internet for the first time or local cache in "tokenizer" folder
    try:
        tokenizer = AutoTokenizer.from_pretrained("tokenizer")
    except Exception as e:
        tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32")
        tokenizer.save_pretrained("tokenizer")

    # Load tf models
    text_encoder_tf = tf.saved_model.load("text_encoder_tf")
    text_infer = text_encoder_tf.signatures["serving_default"]
    vidual_detector_tf = tf.saved_model.load("visual_detector_tf")
    visual_infer = vidual_detector_tf.signatures["serving_default"]
    
    time_stamps[1] = time.time()

    ### Inference 
    ### text -> tokenizer -> text encoding -> text feature
    ### image -> preprocess -> visual input
    ### text feature + visual input -> visual inference -> initial pred: 3 levels of (boxes, scores)
    ### initial pred -> decoding -> filter(NMS) -> final pred: boxes, scores 

    # tokenize texts
    # this process does not depend on learnable parameters
    text_tokens = tokenizer(
        texts,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=77
    )
    
    # text encoding -> output [num_classes, 512]
    text_output = text_infer(
        input_ids=text_tokens["input_ids"],
        attention_mask=text_tokens["attention_mask"]
    )

    time_stamps[2] = time.time()

    print("--------------------------------")
    for k, v in text_output.items():
        print(f"{k}: shape={v.shape}")
        text_feat = v.numpy()
    print("--------------------------------")
    text_feats = text_feat[np.newaxis, :, :]
    text_masks = np.ones((1, text_feats.shape[1]), dtype=np.float32)

    # load and preprocess image
    image_bgr = cv.imread(image_path)
    image_rgb = cv.cvtColor(image_bgr, cv.COLOR_BGR2RGB)
    image_rgb_float = image_rgb.astype(np.float32) / 255.0
    H_orig, W_orig = image_rgb.shape[:2]
    image_resized = cv.resize(image_rgb_float, (640, 640))
    image_tensor = np.transpose(image_resized, (2, 0, 1))[np.newaxis, ...].astype(np.float32)

    # visual inference
    initial_pred = visual_infer(
        image=tf.convert_to_tensor(image_tensor),
        text_feats=tf.convert_to_tensor(text_feats),
        txt_masks=tf.convert_to_tensor(text_masks)
    )
    time_stamps[3] = time.time()
    print("--------------------------------")
    for k, v in initial_pred.items():
        print(f"{k}: shape={v.shape}")
    print("--------------------------------")

    # decode 
    all_boxes, all_cls_probs = [], []

    for level in range(3):
        cls_score = initial_pred[f"cls_score_{level}"].numpy()
        bbox_pred = initial_pred[f"bbox_pred_{level}"].numpy()

        cls_score = torch.from_numpy(cls_score).float()
        bbox_pred = torch.from_numpy(bbox_pred).float()

        cls_prob = torch.sigmoid(cls_score)[0].permute(1, 2, 0).reshape(-1, cls_score.shape[1])
        bbox_pred = bbox_pred[0]  # shape: [4, H, W]
        tx, ty, tr, tb = bbox_pred[0], bbox_pred[1], bbox_pred[2], bbox_pred[3]

        H, W = tx.shape
        stride = 640 / H

        grid_y, grid_x = torch.meshgrid(torch.arange(H), torch.arange(W), indexing='ij')
        x_center = (grid_x + 0.5) * stride
        y_center = (grid_y + 0.5) * stride

        x1 = x_center - tx * stride
        y1 = y_center - ty * stride
        x2 = x_center + tr * stride
        y2 = y_center + tb * stride

        boxes = torch.stack([x1, y1, x2, y2], dim=-1).reshape(-1, 4)
        all_boxes.append(boxes)
        all_cls_probs.append(cls_prob)

    all_boxes = torch.cat(all_boxes, dim=0)
    all_cls_probs = torch.cat(all_cls_probs, dim=0)
    # boxes [8400,4] for 8400 anchors 
    # cls_probs [8400, num_classes]
    print("-------- Boxes and Scores --------")
    print(f"Boxes shape: {all_boxes.shape}")
    print(f"Scores shape: {all_cls_probs.shape}")
    print("--------------------------------")

    scores_list = []
    labels_list = []
    boxes_list = []
    
    for cls_id in range(len(texts)):
        cls_scores = all_cls_probs[:, cls_id]
        labels = torch.ones(cls_scores.shape[0], dtype=torch.long) * cls_id
        keep_idxs = nms(all_boxes, cls_scores, iou_threshold=nms_threshold)
        cur_boxes = all_boxes[keep_idxs]
        cur_scores = cls_scores[keep_idxs]
        cur_labels = labels[keep_idxs]
        
        scores_list.append(cur_scores)
        labels_list.append(cur_labels)
        boxes_list.append(cur_boxes)

    if len(scores_list) > 0:
        final_scores = torch.cat(scores_list, dim=0)
        final_labels = torch.cat(labels_list, dim=0)
        final_boxes = torch.cat(boxes_list, dim=0)
        
        score_mask = final_scores > score_threshold
        final_scores = final_scores[score_mask]
        final_labels = final_labels[score_mask]
        final_boxes = final_boxes[score_mask]
        
        if final_scores.shape[0] > 0:
            keep = nms(final_boxes, final_scores, nms_threshold)
            
            max_dets = 300
            if keep.shape[0] > max_dets:
                keep = keep[:max_dets]
                
            final_boxes = final_boxes[keep]
            final_scores = final_scores[keep]
            final_labels = final_labels[keep]
            
            conf_mask = final_scores > confidence_threshold
            final_boxes = final_boxes[conf_mask]
            final_scores = final_scores[conf_mask]
            final_labels = final_labels[conf_mask]
        else:
            final_boxes = torch.empty((0, 4))
            final_scores = torch.empty((0,))
            final_labels = torch.empty((0,), dtype=torch.long)
    else:
        final_boxes = torch.empty((0, 4))
        final_scores = torch.empty((0,))
        final_labels = torch.empty((0,), dtype=torch.long)
    

    print(f"clases: ")
    for i, cls in enumerate(texts):
        print(f"  {i}: {cls}")

    print("--------------------------------")
    print(f"Final detections: {final_boxes.shape[0]} boxes")

    time_stamps[4] = time.time()

    # bbox
    scale_x, scale_y = W_orig / 640, H_orig / 640
    scaled_boxes = final_boxes.clone()
    scaled_boxes[:, [0, 2]] *= scale_x
    scaled_boxes[:, [1, 3]] *= scale_y

    print(f"detected:")
    vis_image = image_rgb.copy()
    for i in range(final_boxes.shape[0]):
        x1, y1, x2, y2 = map(int, scaled_boxes[i])
        cls_id = final_labels[i].item()
        score = final_scores[i].item()
        label_text = texts[cls_id]
        print(f"  {i}: {label_text} (score: {score:.2f})")
        cv.rectangle(vis_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        font_scale = max(0.5, min(W_orig, H_orig) / 800 * 1.0)
        thickness = max(1, int(min(W_orig, H_orig) / 400))
        label_with_score = f"{cls_id} {score:.2f}"
        (text_w, text_h), baseline = cv.getTextSize(label_with_score,
                                                cv.FONT_HERSHEY_SIMPLEX,
                                                font_scale, thickness)
        rect_y1 = max(y1 - text_h - baseline - 4, 0)
        rect_y2 = max(y1, rect_y1 + text_h + baseline + 4)
        cv.putText(vis_image, label_with_score, (x1 + 2, rect_y2 - baseline - 2),
               cv.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness + 3, cv.LINE_AA)
        cv.putText(vis_image, label_with_score, (x1 + 2, rect_y2 - baseline - 2),
               cv.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness, cv.LINE_AA)

    cv.imwrite(output_path, cv.cvtColor(vis_image, cv.COLOR_RGB2BGR))
    time_stamps[5] = time.time()

    # time cost
    time_stats = {
        "model_loading": time_stamps[1] - time_stamps[0],
        "text_inference": time_stamps[2] - time_stamps[1],
        "visual_inference": time_stamps[3] - time_stamps[2],
        "postprocess": time_stamps[4] - time_stamps[3],
        "save": time_stamps[5] - time_stamps[4],
        "total": time_stamps[5] - time_stamps[0]
    }
    print("------------------------------")
    for name, time_cost in time_stats.items():
        print(f"{name.title()}: {time_cost:.3f}s")
    print("------------------------------")

    plt.figure(figsize=(12, 10))
    plt.imshow(vis_image)
    plt.axis("off")
    plt.title("YOLO-World Detection Result (TFLite)")
    plt.show()
    
    return {
        "boxes": final_boxes,
        "scores": final_scores,
        "labels": final_labels,
        "class_names": texts,
        "output_path": output_path,
        "time_stats": time_stats,
        "visualization": vis_image
    }

if __name__ == "__main__":
    test_texts = ["bottle"]
    test_image = "sample_images/bottles.png"
    
    results = yolo_world_detect(test_texts,
                               test_image,
                               output_path="output.jpg",
                               score_threshold=0.05,
                               nms_threshold=0.5,
                               confidence_threshold=0.25
                              )

