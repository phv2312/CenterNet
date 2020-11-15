import json
from pycocotools.coco import COCO
import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
import pylab
import cv2

image_directory   = "/home/kan/Desktop/cinnamon/zalo/CenterNet/data/coco_zalo/images/"
lbl_path = "/home/kan/Desktop/cinnamon/zalo/CenterNet/data/coco_zalo/train.json"

json_data = json.load(open(lbl_path, 'r'))


example_coco = COCO(lbl_path)

categories = example_coco.loadCats(example_coco.getCatIds())
category_names = [category['name'] for category in categories]
print('Custom COCO categories: \n{}\n'.format(' '.join(category_names)))

category_names = set([category['supercategory'] for category in categories])
print('Custom COCO supercategories: \n{}'.format(' '.join(category_names)))

category_ids = []
image_ids = example_coco.getImgIds(catIds=category_ids)

while(True):
    image_data = example_coco.loadImgs(image_ids[np.random.randint(0, len(image_ids))])[0]



    # load and display instance annotations
    image = io.imread(image_directory + image_data['file_name'])
    # plt.imshow(image)
    # plt.show()
    pylab.rcParams['figure.figsize'] = (8.0, 10.0)
    annotation_ids = example_coco.getAnnIds(imgIds=image_data['id'], catIds=category_ids, iscrowd=None)
    annotations = example_coco.loadAnns(annotation_ids)

    for annot in annotations:
        x_min, y_min, w, h = annot['bbox']

        x_max = x_min + w
        y_max = y_min + h
        print(annot)

        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (255,0,0), thickness=1)

    #example_coco.showAnns(annotations, draw_bbox=True)
    plt.imshow(image)
    plt.show()