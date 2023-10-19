import os
from groundingdino.util.inference import load_model, load_image, predict, annotate
import cv2

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
CONFIG_PATH = "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
WEIGHTS_PATH = os.path.join("weights", "groundingdino_swint_ogc.pth")
model = load_model(CONFIG_PATH, WEIGHTS_PATH)
procfolde = "test_set"


def find_from_image(image_pth, prompt, bt, tt):
    text_prompt = prompt
    bt = bt
    tt = tt
    imgsrc, img = load_image(image_pth)
    boxes, logits, phrases = predict(
        model=model, image=img, caption=text_prompt, box_threshold=bt, text_threshold=tt
    )
    annotated_image = annotate(imgsrc, boxes, logits, phrases)
    cv2.imwrite(f"results/{os.path.basename(image_pth)}.png", annotated_image)
    return boxes.tolist()


for file in os.listdir(procfolde):
    img = os.path.join(procfolde, file)
    result = find_from_image(image_pth=img, prompt="human head", bt=0.5, tt=0.22)
    print(result)
