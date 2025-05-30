# YOLO-World TFLite Inference

## Models

### text_encoder_tf
- **Input:** text token
- **Output:** text_feature

### visual_detector_tf
- **Input:** text feature and image data
- **Output:** initial prediction

## Directory Structure

```
.
├─sample_images/        # Test images
├─text_encoder_tf/      # Text encoder model
├─tokenizer/           # CLIP tokenizer files
└─visual_detector_tf/   # Visual detection model
```

## inference process
- text -> tokenizer -> text encoding -> text feature
- image -> preprocess -> visual input
- text feature + visual input -> visual inference -> initial pred: 3 levels of (boxes, scores)
- initial pred -> decoding -> filter(NMS) -> final pred: boxes, scores 

## Usage

```python
test_texts = ["champagne bottle", "red bottle", "plastic spoon"]
test_image = "sample_images/desk_test.png"

results = yolo_world_detect(test_texts,
                             test_image,
                             output_path="output.jpg",
                             score_threshold=0.05,
                             nms_threshold=0.5,
                             confidence_threshold=0.25
                            )
```

## Parameters

| Parameter   | Type        | Description                     | Default      |
|-------------|-------------|---------------------------------|--------------|
| `texts`     | `List[str]` | List of detection target texts  | Required     |
| `image_path`| `str`       | Input image path                | Required     |
| `min_thresh`| `float`     | Original score threshold        | 0.05         |
| `norm_thresh`| `float`    | Normalized score threshold      | 0.85         |
| `output_path`| `str`      | Output image save path          | "output.jpg" |

## Output

The function returns a dictionary containing:

### Detection Results
- `boxes`: Bounding box coordinates
- `scores`: Detection confidence scores
- `labels`: Class labels
- `names`: Class names

### Additional Information
- `output_path`: Path to saved output image
- `time_stats`: Inference time statistics
- `visualization`: Visualization result image

## Technical Notes

### CLIP Tokenizer
```
Recommendation: Use Hugging Face's tokenizers API (C++ implementation)
```

### Image Preprocessing
Requirements:
- Resolution: 640x640
- Normalization: [0, 1]
- Format conversion: HWC RGB → CHW

### NMS Implementation
Current status:
- Uses PyTorch's `torchvision.ops.nms`
- Not suitable for Android deployment
- Need custom implementation for mobile

Reference implementation:
- Source: `YOLO-World/deploy/easydeploy/nms`
- Official TFLite demo: `YOLO-World/deploy/tflite_demo.py`
