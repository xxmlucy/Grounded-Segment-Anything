from groundingdino.util.inference import load_model, load_image, predict, annotate
import cv2
import torch

# Set paths
CONFIG_PATH = "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
CHECKPOINT_PATH = "./groundingdino_swint_ogc.pth"
IMAGE_PATH = "assets/demo7.jpg"

# Set parameters
TEXT_PROMPT = "Horse. Clouds. Grasses. Sky. Hill."
BOX_THRESHOLD = 0.35
TEXT_THRESHOLD = 0.25

# Determine device
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
FP16_INFERENCE = DEVICE.type == 'cuda'

# Load image and model
image_source, image = load_image(IMAGE_PATH)
model = load_model(CONFIG_PATH, CHECKPOINT_PATH)

# Use FP16 if CUDA is available
if FP16_INFERENCE:
    image = image.half()
    model = model.half()

# Move model to the device
model.to(DEVICE)

# Make prediction
boxes, logits, phrases = predict(
    model=model,
    image=image,
    caption=TEXT_PROMPT,
    box_threshold=BOX_THRESHOLD,
    text_threshold=TEXT_THRESHOLD,
    device=DEVICE,
)

# Annotate and save the image
annotated_frame = annotate(image_source=image_source, boxes=boxes, logits=logits, phrases=phrases)
cv2.imwrite("annotated_image.jpg", annotated_frame)
