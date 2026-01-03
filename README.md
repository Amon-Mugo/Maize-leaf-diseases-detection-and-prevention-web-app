Maize Leaf Disease Detection with OOD Awareness

This project is a deep learning–based maize leaf disease detection system enhanced with Out-of-Distribution (OOD) detection to ensure reliable and safe predictions during real-world deployment.

It classifies maize leaf images into known disease categories and rejects unknown inputs (e.g., non-leaf images) using energy-based OOD detection with temperature scaling.

🚀 Features

✅ Maize disease classification

✅ Out-of-Distribution (OOD) detection

✅ Confidence scoring

✅ Preventive recommendations per disease

✅ FastAPI backend

✅ Modern web interface with image preview

✅ Production-ready inference (no retraining required)

🧠 Model Overview

Backbone: ResNet-50

Training strategy:

Stage-wise fine-tuning

Frozen backbone → fine-tuned last layers

Early stopping & LR scheduler

Calibration:

Temperature Scaling (T = 0.5047)

OOD Method:

Energy-based OOD detection

Threshold calibrated at 95th percentile of ID validation data

🏷️ Supported Classes
Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot
Corn_(maize)___Common_rust_
Corn_(maize)___Northern_Leaf_Blight
Corn_(maize)___healthy

🧪 How OOD Detection Works

The model outputs logits

Energy score is computed:

E(x)=−T⋅log⁡∑exp⁡(logitsT)
E(x)=−T⋅log∑exp(
T
logits
	​

)

If energy > calibrated threshold → image is rejected as Unknown (OOD)

This prevents the model from making confident but incorrect predictions on unrelated images.

🖥️ Web Application Demo
✅ In-Distribution (Correct Disease Detection)

The model correctly identifies a maize leaf disease, returns the class name, confidence score, and a preventive recommendation.

Example Output:

Disease: Cercospora Leaf Spot

Confidence: 98.47%

Recommendation: Use resistant varieties and remove infected debris.

🚫 Out-of-Distribution Detection (OOD)

When a non-maize image (e.g., a car) is uploaded, the system rejects the input instead of guessing.

Result:

Prediction: Unknown (OOD)

Message: No matching class detected.

This behavior is critical for safe real-world deployment.

🛠️ Tech Stack

Python

PyTorch

TorchVision

FastAPI

Jinja2 (Templates)

HTML / CSS / JavaScript

Pop!_OS (Linux)

▶️ Running the Project Locally
1️⃣ Install dependencies
pip install torch torchvision fastapi uvicorn python-multipart pillow

2️⃣ Start the server
uvicorn app2:app --reload

3️⃣ Open in browser
http://127.0.0.1:8000

📁 Project Structure
├── app2.py                  # FastAPI application
├── resnet50_maize_ood.pth   # Model + temperature + OOD threshold
├── templates/
│   └── index_new.html       # Frontend UI
├── static/
│   └── style_new.css        # Styling
├── screenshots/
│   ├── id_prediction.png
│   └── ood_prediction.png
└── README.md

🎯 Why OOD Detection Matters

Without OOD detection:

Models always predict, even on nonsense inputs

With OOD detection:

The system knows when it doesn’t know

Prevents harmful or misleading decisions

Essential for agriculture, healthcare, and real deployments

👨‍🎓 Author

Amon
3rd Year University Student
Focus: Machine Learning, Data Science, and Model Deployment
