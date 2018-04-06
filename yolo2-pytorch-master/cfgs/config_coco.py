import numpy as np
from pycocotools.coco import COCO
# #
coco_obj = COCO('/home/kwx/OBJ2TEXT-Improved/data/annotations/instances_train2014.json')
#

label_to_ind = {}
i = 0
for key in coco_obj.cats:
    label_to_ind[key] = i
    i+=1
#
name_to_label = {}
for label in coco_obj.cats:
    name_to_label[coco_obj.cats[key]['name']] = label


name_to_ind = {}
i = 0
for label in coco_obj.cats:
    name_to_label[coco_obj.cats[key]['name']] = i
    i+=1


# trained model
h5_fname = 'yolo-voc.weights.h5'

# COCO
label_names = ('person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush')

num_classes = len(label_names)


anchors = np.asarray([(1.08, 1.19), (3.42, 4.41),
                      (6.63, 11.38), (9.42, 5.11), (16.62, 10.52)],
                     dtype=np.float)
num_anchors = len(anchors)


