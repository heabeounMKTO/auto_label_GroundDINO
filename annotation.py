import json
import os
import numpy as np
import cv2
import torch
from pprint import pprint


class Label:
    def __init__(self, coords, cls, image, filename):
        self.coords = coords
        self.cls = cls
        self.img_w = image.shape[1]
        self.img_h = image.shape[0]
        self.fname = os.path.basename(filename)

    def __str__(self):
        return f"coords: {self.coords} , cls {self.cls}, w,h {self.img_w, self.img_h}"

    def create_label(self):
        shapes = []
        for index, cord in enumerate(self.coords):
            orig_coords = self.real_coords(cord)
            points0 = [orig_coords[0], orig_coords[1]]
            points1 = [orig_coords[2], orig_coords[3]]
            points = {
                "label": (self.cls[index]).replace(" ", "_"),
                "points": [points0, points1],
                "group_id": "null",
                "shape_type": "rectangle",
                "flags": {},
            }
            shapes.append(points)
        label_fmt = {
            "version": "5.1.1",
            "flags": {},
            "shapes": shapes,
            "imagePath": self.fname,
            "imageHeight": self.img_h,
            "imageWidth": self.img_w,
        }
        return label_fmt

    def real_coords(self, x, padw=0, padh=0):
        """
        converts normalzied coords to image coords
        """
        y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
        y[0] = self.img_w * (x[0] - x[2] / 2) + padw  # top left x
        y[1] = self.img_h * (x[1] - x[3] / 2) + padh  # top left y
        y[2] = self.img_w * (x[0] + x[2] / 2) + padw  # bottom right x
        y[3] = self.img_h * (x[1] + x[3] / 2) + padh  # bottom right y
        y = [int(x) for x in y.tolist()]
        return y
