from annotation import Label
from dino_finder import DinoFinder
import os
import cv2
import json

TEST_IMG = "test_set/t1.jpeg"
cv_TEST_IMG = cv2.imread(TEST_IMG)
CONFIG_PATH = "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py" 
WEIGHTS_PATH = os.path.join("weights", "groundingdino_swint_ogc.pth")
finder = DinoFinder(CONFIG_PATH, WEIGHTS_PATH)

result_bbox, _conf, _cls = finder.find_by_prompt("human head. tree", TEST_IMG)
print(result_bbox, _conf, _cls)

labelz = Label(result_bbox, _cls, cv_TEST_IMG, TEST_IMG).create_label()
jsonFilename = os.path.splitext(TEST_IMG)[0] + ".json"
print(labelz)
with open(jsonFilename, "w") as json_output:
    json_output.write(json.dumps(labelz, indent=2))