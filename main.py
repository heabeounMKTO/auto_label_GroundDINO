from annotation import Label
from dino_finder import DinoFinder
import os
import cv2
import json
from tqdm import tqdm

TEST_IMG = "test_set/t1.jpeg"
cv_TEST_IMG = cv2.imread(TEST_IMG)
CONFIG_PATH = "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py" 
WEIGHTS_PATH = os.path.join("weights", "groundingdino_swint_ogc.pth")
finder = DinoFinder(CONFIG_PATH, WEIGHTS_PATH)
workfolder = "/home/hbpopos/gtadset/sorted_dset"

for image in tqdm(os.listdir(workfolder)):
    if image.endswith(".jpeg"):
        image_path = os.path.join(workfolder, image)
        result_bbox, _conf, _cls = finder.find_by_prompt("human head", image_path)
        labelz = Label(result_bbox, _cls, cv_TEST_IMG, image_path).create_label()
        jsonFilename = os.path.splitext(image_path)[0] + ".json"
        # print(labelz)
        with open(jsonFilename, "w") as json_output:
            json_output.write(json.dumps(labelz, indent=2))