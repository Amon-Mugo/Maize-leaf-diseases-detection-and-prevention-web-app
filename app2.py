from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from PIL import Image
import io
import torch
from torchvision import transforms

# -------------------------
# Load Model + Config
# -------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load saved OOD model checkpoint
checkpoint = torch.load("resnet50_maize_ood.pth", map_location=device)

# Classes
CLASS_NAMES = [
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
    "Corn_(maize)___Common_rust_",
    "Corn_(maize)___Northern_Leaf_Blight",
    "Corn_(maize)___healthy"
]

# Recreate model architecture
import torchvision.models as models
model = models.resnet50(weights=None)
model.fc = torch.nn.Sequential(
    torch.nn.BatchNorm1d(2048),
    torch.nn.ReLU(inplace=True),
    torch.nn.Linear(2048, len(CLASS_NAMES))
)
model.load_state_dict(checkpoint['model_state_dict'])
model.to(device)
model.eval()

T = checkpoint['temperature']
ood_threshold = checkpoint['ood_threshold']

# -------------------------
# App + Templates
# -------------------------
app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# -------------------------
# Image Transform
# -------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# -------------------------
# OOD + Prediction Logic
# -------------------------
def energy_score(logits, T):
    return -T * torch.logsumexp(logits / T, dim=1)

def predict_with_ood(img_tensor):
    img_tensor = img_tensor.to(device).unsqueeze(0)
    with torch.no_grad():
        logits = model(img_tensor)
        energy = energy_score(logits, T)
        if energy.item() > ood_threshold:
            return {"prediction": "Unknown", "energy": energy.item(), "tips": "No matching class. Please check leaf or upload a clear image."}
        else:
            probs = torch.softmax(logits / T, dim=1)
            class_idx = probs.argmax(dim=1).item()

            # Preventive tips for each disease
            tips_dict = {
                "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot": "Use resistant varieties and remove infected debris.",
                "Corn_(maize)___Common_rust_": "Apply recommended fungicides and rotate crops.",
                "Corn_(maize)___Northern_Leaf_Blight": "Ensure proper fertilization and avoid dense planting.",
                "Corn_(maize)___healthy": "Plant is healthy. Maintain good field hygiene."
            }

            return {
                "prediction": CLASS_NAMES[class_idx],
                "confidence": probs[0, class_idx].item(),
                "energy": energy.item(),
                "tips": tips_dict[CLASS_NAMES[class_idx]]
            }

# -------------------------
# Routes
# -------------------------
@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index_new.html", {"request": request})

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        img = Image.open(io.BytesIO(contents)).convert("RGB")
        img_tensor = transform(img)
        result = predict_with_ood(img_tensor)
        return JSONResponse(content=result)
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=400)

