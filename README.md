🔍 Object Detection and Classification using EfficientDet + MobileNetV2 (Colab)
This project enables real-time or static object detection and classification using a combination of EfficientDet (Lite2) for detecting multiple objects and MobileNetV2 for classifying each detected object. Designed specifically for Google Colab, this implementation is optimized for low memory usage and avoids session crashes caused by large images or excessive computations.

🚀 Features
✅ Upload or capture image directly in Colab

🧠 EfficientDet Lite2 for fast and accurate object detection

🎯 MobileNetV2 for classifying detected objects (lightweight and fast)

🧹 Safe preprocessing (resizes input images to 320×320 to avoid crashes)

🔒 Memory-safe: limits object count and avoids rendering overload

📦 All-in-one Colab-compatible code (no external setup required)

📷 Demo
<p align="center"> <img src="https://user-images.githubusercontent.com/example/detection_classification.png" alt="demo" width="500"/> </p>
🧰 Technologies Used
Python 3.9+

TensorFlow 2.x

TensorFlow Hub

MobileNetV2 (ImageNet pre-trained)

EfficientDet (Lite2 variant from TF Hub)

Google Colab UI (for webcam & file uploads)

📁 How to Use (in Colab)
Open the notebook in Google Colab.

Run the cells to install dependencies.

Choose one:

📁 Upload an image (u)

📸 Capture an image via webcam (c)

The model will detect and classify up to 3 objects.

Output is printed directly in the notebook.

🔐 Notes
For high-resolution images, the input is downscaled to 320x320 to prevent Colab session crashes.

The number of detected objects is capped at 3 to reduce memory usage.

The classifier used is MobileNetV2 for faster inference and lower RAM consumption.

📌 To Do (Optional Extensions)
 Add YOLOv8-based object detection

 Export predictions to JSON/CSV

 Draw and save annotated output image

 Enable batch processing for multiple images

🤝 Credits
EfficientDet Lite2: TF Hub Model

MobileNetV2: Keras Applications

COCO labels: amikelive/coco-labels

