# Step 1: Install and import libraries
!pip install -q tensorflow seaborn opencv-python-headless

import tensorflow as tf
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from google.colab import files
from IPython.display import display, Javascript
from google.colab.output import eval_js
from base64 import b64decode

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import io

# Step 2: Load VGG16 model
model = VGG16(weights='imagenet')

# Step 3: Webcam capture function (for Colab)
def capture_from_webcam(filename='captured.jpg'):
    js = Javascript('''
    async function takePhoto() {
      const div = document.createElement('div');
      const capture = document.createElement('button');
      capture.textContent = '📸 Capture';
      div.appendChild(capture);

      const video = document.createElement('video');
      video.style.display = 'block';
      const stream = await navigator.mediaDevices.getUserMedia({video: true});

      document.body.appendChild(div);
      div.appendChild(video);
      video.srcObject = stream;
      await video.play();

      google.colab.output.setIframeHeight(document.documentElement.scrollHeight, true);
      await new Promise((resolve) => capture.onclick = resolve);

      const canvas = document.createElement('canvas');
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
      canvas.getContext('2d').drawImage(video, 0, 0);
      stream.getTracks().forEach(track => track.stop());
      div.remove();
      return canvas.toDataURL('image/jpeg', 0.8);
    }
    takePhoto();
    ''')
    display(js)
    data = eval_js("takePhoto()")
    binary = b64decode(data.split(',')[1])
    with open(filename, 'wb') as f:
        f.write(binary)
    return filename

# Step 4: Ask for input method
choice = input("Type 'u' to upload an image or 'c' to capture from webcam: ").lower()
image_files = []

if choice == 'u':
    uploaded = files.upload()
    image_files = list(uploaded.keys())

elif choice == 'c':
    img_path = capture_from_webcam()
    image_files = [img_path]

else:
    print("Invalid input. Please type 'u' or 'c'.")

# Step 5: Predict and display
TOP_N = 5

for filename in image_files:
    img = load_img(filename, target_size=(224, 224))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    preds = model.predict(img_array)
    decoded = decode_predictions(preds, top=TOP_N)[0]

    labels = [label for (_, label, _) in decoded]
    confidences = [round(prob * 100, 2) for (_, _, prob) in decoded]

    # Show image with title
    plt.figure(figsize=(6, 6))
    plt.imshow(Image.open(filename))
    plt.axis('off')
    plt.title(f"Top Prediction: {labels[0]} ({confidences[0]}%)")
    plt.show()

    # Bar plot
    sns.barplot(x=confidences, y=labels, palette='viridis')
    plt.xlabel("Confidence (%)")
    plt.title("Top Predictions")
    plt.xlim(0, 100)
    plt.show()

    # Print in text
    print(f"Predictions for: {filename}")
    for i in range(TOP_N):
        print(f"{i+1}. {labels[i]} ({confidences[i]}%)")
