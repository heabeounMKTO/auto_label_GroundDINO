import torch
from segment_anything import SamPredictor,SamAutomaticMaskGenerator,sam_model_registry
import cv2

class SamMasker:
    def __init__(self, device, weights, model_type="vit_h"):
        self.model = sam_model_registry[model_type](checkpoint=weights)
        self.model.to(device=device)
        self.mask_gen = SamAutomaticMaskGenerator(self.model) 
        self.pred = SamPredictor(self.model)
    def sam_mask_from_img(self, img_arr):
        return self.mask_gen.generate(img_arr)

    def sam_pred_from_img(self, img_arr, box):
        self.pred.set_image(img_arr)
        masks, scores, logits = self.pred.predict(
            box = box,
            multimask_output=True
        )
        return masks, scores, logits 