# Drone-Based-AI-System-For-Date-Palm-Health-Monitoring-in-the-UAE
Senior Design Project: A drone-based AI system for intelligent palm tree monitoring in the UAE, integrating YOLO-based detection and counting, EfficientNet-based health and disease classification, and U-Net-based disease severity segmentation as an end-to-end system for intelligent palm tree health assessment.


# Model Skeleton
## Input

- Images or video (video is split into frames)
- All inputs are treated as RGB images

## Stage 1: Object Detection (YOLO)

- Identify palm trees in the image
- Count the palm trees
- Output the ROI (Region of Interest) for each tree
- Crop each ROI to pass to the next stage

## Stage 2: Multi-class Image Classification (EfficientNet)

- Takes the cropped palm tree images from Stage 1
- Classifies each crop into one of several classes:
  - Healthy
  - Disease/problem types (multiple classes)

## Stage 3: Segmentation (EfficientNet U-Net)

- Takes the crops classified as diseased/problematic
- Segments the affected regions
- Identifies the **severity** of the problem

## Final Output (per image)

- Total palm tree count
- Health or disease class for each tree
- Severity level for each affected tree
