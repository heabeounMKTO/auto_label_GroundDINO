import gradio as gr
from dino_finder import DinoFinder
from sam_masker import SamMasker
import os
import cv2
from PIL import Image
import torch
import numpy as np
import supervision as sv

CONFIG_PATH = "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
WEIGHTS_PATH = os.path.join("weights", "groundingdino_swint_ogc.pth")
SAM_WEIGHTS = "weights/sam_vit_h_4b8939.pth"

finder = DinoFinder(CONFIG_PATH, WEIGHTS_PATH)
masker = SamMasker('cuda:0', SAM_WEIGHTS)

mask_annotator = sv.MaskAnnotator()


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
        pil_img, coord, _crop_img = crop_image(box, img)
        col.paste(pil_img, coord)
    return col

def show_mask(mask,  random_color=True):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    return mask_image

def crop_image(box, img):
    rc = real_coords(np.array(box), img).tolist()
    rc = [int(x) for x in rc]
    crop_img = img[rc[1] : rc[3], rc[0] : rc[2]]
    pil_img = Image.fromarray(crop_img)
    return pil_img, (rc[0],rc[1]),crop_img

def filter_result_by_prompt(prompt, img):
    binary_img = None
    all_binary_mask = np.zeros([img.shape[0], img.shape[1]], dtype=np.uint8) 
    for idx, box in enumerate(finder.boxes):
        if finder.cls[idx] == prompt:
            pil_img, coord, crop_img = crop_image(np.array(box), img) 
            sam_res, scores, logit = masker.sam_pred_from_img(img, np.array([int(x) for x in real_coords(np.array(box), img)]))
            detections = sv.Detections(
                xyxy=sv.mask_to_xyxy(masks=sam_res),
                mask=sam_res
            )
            detections = detections[detections.area == np.max(detections.area)]
            anno_img = mask_annotator.annotate(scene=img.copy(), detections=detections)
            highest_score_idx = int(np.where(scores == np.max(scores))[0])
            binary_img = np.array(sam_res[highest_score_idx], dtype=np.uint8)
            binary_img[binary_img > 0] = 255
            all_binary_mask += binary_img        
    masked_img = cv2.bitwise_and(img, img, mask=all_binary_mask) 
    return masked_img, all_binary_mask 

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
            filter_box = gr.Textbox(label="please enter a single (1) class")
            filter_img = gr.Button("Filter Results") 
            filter_res = gr.Image()
            filter_res_bin = gr.Image()
    greet_button.click(
        fn=find_from_prompt,
        inputs=[imginput, prompt, confidence, text_confidence],
        outputs=[imgoutput, collage],
        api_name="find from prompt",
    )
    filter_img.click(
        fn = filter_result_by_prompt,
        inputs=[filter_box, imginput],
        outputs= [filter_res, filter_res_bin],
        api_name="filter results"    
    )

demo.launch(share=True)
