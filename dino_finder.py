import os 
import json 
import cv2 
from groundingdino.util.inference import load_model, load_image, predict, annotate

class DinoFinder():
    def __init__(self,config_path,weights_path):
        os.environ["CUDA_LAUNCH_BLOCKING"] = "1" # need this to run cos somehow the thing runs async ? (brain too small)
        self.model = load_model(config_path, weights_path)
    def find_by_prompt(self, prompt, image_path, bt=0.55, tt=0.25, save=False):
        original_img , img = load_image(image_path) 
        boxes, logits, phrases = predict(
            model=self.model,
            image=img,
            caption=prompt,
            box_threshold=bt,
            text_threshold=tt  
        )
        if save == True:
            if os.path.exists("results") == False:
                os.makedirs("results")
            cv2.imwrite(f"results/{os.path.basename(image_path)}", annotate(original_img, boxes, logits, phrases))
        return boxes.tolist()