# YOLO-World TFLite Inference Demo

## Overview
This demo shows how to perform object detection using the TFLite version of YOLO-World model. It supports multi-object detection with custom text prompts.

**Important Notes:**
- The export script must be run from YOLO-World root directory to access custom model components
- One text prompt at a time
- One image at a time

## Pre-trained Models
The pre-trained model weights can be found in [YOLO-World/configs/pretrain_v1/README.md] or [./pretrained_v1.md]

## Directory Structure
```
demo/TFLite_demo/
├── export.py               # Model export script
├── TFLite_inference.py    # TFLite inference implementation
├── README.md              # Documentation
├── sample_images/         # Example images
│   ├── bottles.png
│   ├── bus.jpg
│   ├── desk.png
│   └── ...
└── tokenizer/            # CLIP tokenizer files
    ├── merges.txt
    ├── tokenizer.json
    ├── vocab.json
    └── ...
```

## Model Components

### Text Encoder
- Input: Text tokens
- Output: Text features
- Location: `text_encoder_tf/` directory

### Visual Detector
- Input: Text features and image data
- Output: Initial predictions
- Location: `visual_detector_tf/` directory

## Inference Pipeline
1. Text Processing: Text → Tokenizer → Text Encoding → Text Features
2. Image Processing: Image → Preprocessing → Visual Input
3. Model Inference: Text Features + Visual Input → Visual Inference → Initial Predictions (3 levels of boxes and scores)
4. Post-processing: Initial Predictions → Decoding → NMS Filtering → Final Predictions

## Usage Example
```python
# Define single detection target and single input image
test_texts = ["champagne bottle"]  # Only one text prompt supported
test_image = "sample_images/desk_test.png"  # Single image input

# Perform detection
results = yolo_world_detect(
    texts=test_texts,
    image_path=test_image,
    output_path="output.jpg",
    score_threshold=0.05,    # Raw score threshold
    nms_threshold=0.5,       # NMS threshold
    confidence_threshold=0.25 # Confidence threshold
)
```

## Parameters
| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `texts` | `List[str]` | List of target texts for detection | Required |
| `image_path` | `str` | Input image path | Required |
| `output_path` | `str` | Output image save path | "output.jpg" |
| `score_threshold` | `float` | Detection score threshold | 0.05 |
| `nms_threshold` | `float` | NMS threshold | 0.5 |
| `confidence_threshold` | `float` | Confidence threshold | 0.25 |

## Return Value
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
1. Image Preprocessing Requirements:
   - Resolution: 640x640
   - Normalization: [0, 1]
   - Format conversion: HWC RGB → CHW

2. NMS Implementation:
   - Currently using PyTorch's torchvision.ops.nms
   - Custom implementation needed for mobile deployment
   - Reference implementations:
     - YOLO-World/deploy/easydeploy/nms
     - YOLO-World/deploy/tflite_demo.py
