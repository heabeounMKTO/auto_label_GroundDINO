import gradio as gr
from dino_finder import DinoFinder
import os
import cv2
from PIL import Image
import torch
import numpy as np


CONFIG_PATH = "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
WEIGHTS_PATH = os.path.join("weights", "groundingdino_swint_ogc.pth")
finder = DinoFinder(CONFIG_PATH, WEIGHTS_PATH)


def real_coords(x, img):
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    w = img.shape[1]
    h = img.shape[0]

    y[0] = w * (x[0] - x[2] / 2)  # top left x
    y[1] = h * (x[1] - x[3] / 2)  # top left y
    y[2] = w * (x[0] + x[2] / 2)  # bottom right x
    y[3] = h * (x[1] + x[3] / 2)  # bottom right y
    return y


def create_mask_collage(box_arr, cls_arr, img):
    collages = []
    imgc = []
    col = Image.new("RGBA", (img.shape[1], img.shape[0]))
    for box in box_arr:
        rc = real_coords(np.array(box), img).tolist()
        rc = [int(x) for x in rc]
        crop_img = img[rc[1] : rc[3], rc[0] : rc[2]]
        pil_img = Image.fromarray(crop_img)
        col.paste(pil_img, (rc[0], rc[1]))
    return col


def find_from_prompt(image_path, prompt, confidence, text_confidence):
    print(f"PROOMPT : {prompt, type(prompt)}")
    boxes, conf, cls, an = finder.find_by_prompt(
        prompt=prompt,
        bt=(confidence / 100),
        tt=(text_confidence / 100),
        image_path=image_path,
    )
    an = cv2.cvtColor(an, cv2.COLOR_BGR2RGB)
    collage = create_mask_collage(boxes, cls, image_path)
    return an, collage


with gr.Blocks() as demo:
    with gr.Accordion("GroundingDINO and SAM, a prompt-based auto segmentation demo"):
        gr.Markdown("yes")
    with gr.Row():
        with gr.Column():
            imginput = gr.Image()
            prompt = gr.Textbox(label="Prompt")
            confidence = gr.Slider(0, 100, label="confidence", value=50)
            text_confidence = gr.Slider(0, 100, label="text confidence", value=25)
            greet_button = gr.Button("Lessgoooo")
        with gr.Column():
            imgoutput = gr.Image()
            collage = gr.Image()
            # imgm = gr.Image()
        with gr.Column():
            filter_box = gr.Textbox(label="filter")
            filter_img = collage
    greet_button.click(
        fn=find_from_prompt,
        inputs=[imginput, prompt, confidence, text_confidence],
        outputs=[imgoutput, collage],
        api_name="greet",
    )

demo.launch()
