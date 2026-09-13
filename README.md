# Urban Footprint Semantic Segmentation

This project presents a robust deep learning pipeline designed to automatically extract building footprints from high-resolution DeepGlobe satellite imagery. Engineered to support Smart City development, civil engineering, and urban planning, this computer vision system accurately translates unstructured geospatial RGB data into precise, pixel-perfect binary masks.

## Architecture and Tech Stack

- **Deep Learning Framework:** PyTorch utilizing `segmentation-models-pytorch`.
- **Model Topology:** U-Net architecture integrated with a pre-trained ResNet34 encoder to capture complex spatial geometries and feature hierarchies.
- **Data Pipeline:** Albumentations for rigorous spatial augmentations, ensuring model robustness against varying satellite angles, lighting, and orientations.
- **Post-Processing:** OpenCV for deterministic noise filtering and spatial component analysis.

## Overcoming Engineering Challenges

Developing a production-ready vision model required diagnosing and solving several critical mathematical and programmatic obstacles during the training lifecycle:

- **Color Space Distortion:** Standard PyTorch ImageNet normalization heavily distorted the visual output during inference. This was solved by engineering a precise un-normalization tensor operation to restore the natural geospatial RGB values, allowing for accurate visual validation and academic review.
- **False-Positive Hallucinations:** Early iterations consistently hallucinated buildings in completely empty agricultural fields due to mathematical instability when evaluating empty ground-truth masks. This was corrected by implementing a custom, smoothed Dice-BCE loss function, preventing zero-division errors and heavily penalizing erroneous predictions on empty terrain.
- **Sub-Pixel Salt Noise:** To eliminate microscopic artifacts that sneaked past the standard probabilistic confidence thresholds, an OpenCV Connected Component Analysis algorithm was integrated. This module physically calculates the pixel area of predicted spatial islands, strictly deleting topological anomalies below a 50-pixel threshold to ensure clean, actionable output maps.
