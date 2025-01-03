import os
import json
import cv2
from groundingdino.util.inference import load_model, load_image, predict, annotate
import torch
import groundingdino.datasets.transforms as T
from PIL import Image


class DinoFinder:
    def __init__(self, config_path, weights_path):
        os.environ[
            "CUDA_LAUNCH_BLOCKING"
        ] = "1"  # need this to run cos somehow the thing runs async ? (brain too small)
        self.model = load_model(config_path, weights_path)

    def find_by_prompt(self, prompt, image_path, bt=0.55, tt=0.25, save=False):
        if os.path.isfile(image_path):
            original_img, img = load_image(image_path)
        else:
            transform = T.Compose(
                [
                    T.RandomResize([800], max_size=1333),
                    T.ToTensor(),
                    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ]
            )
            original_img = image_path.copy()
            img, _ = transform(Image.fromarray(original_img), None)
            # print("img_type", type(img))
        boxes, logits, phrases = predict(
            model=self.model,
            image=img,
            caption=prompt,
            box_threshold=bt,
            text_threshold=tt,
        )
        anno = annotate(original_img, boxes, logits, phrases)
        if save == True:
            if os.path.exists("results") == False:
                os.makedirs("results")
            cv2.imwrite(f"results/{os.path.basename(image_path)}", anno)
        self.boxes = boxes 
        self.conf = logits
        self.cls = phrases
        return boxes, logits, phrases, anno
