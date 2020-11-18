import sys

CENTERNET_PATH = "./lib"
sys.path.insert(0, CENTERNET_PATH)

import cv2
import os
import glob
import json
from natsort import natsorted
import matplotlib.pyplot as plt
from detectors.detector_factory import detector_factory
from datasets.dataset_factory import get_dataset
from opts import opts

# input path for both model and image path
model_path = "/home/kan/Desktop/coco_zalo_weights/coco_zalo/model_best.pth"
img_path = "/home/kan/Desktop/cinnamon/zalo/CenterNet/data/coco_zalo/images/5.png"
test_data_dir = "/home/kan/Desktop/question/za_traffic_2020/traffic_public_test/images"
lbl_id2name = {
    0: 'background',
    1: 'cam_nguoc_chieu',
    2: 'cam_dung_va_do',
    3: 'cam_re',
    4: 'gioi_han_toc_do',
    5: 'cam_con_lai',
    6: 'nguy_hiem',
    7: 'hieu_lenh'
}

def visulize(image, results, num_classes, vis_thresh):
    # visualize ...
    for j in range(1, num_classes + 1):
        for bbox in results[j]:
            if bbox[4] > vis_thresh:
                cv2.rectangle(image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color=(0, 255, 0), thickness=2)
                cv2.putText(image, lbl_id2name[j], (int(bbox[0]), int(bbox[1]) - 5), cv2.FONT_HERSHEY_TRIPLEX, 1,
                            (255, 0, 0), 2)

    return image

# load model & aruguments
TASK = 'ctdet'  # or 'multi_pose' for human pose estimation
opt = opts().init('{} --load_model {}'.format(TASK, model_path).split(' '))
Dataset = get_dataset('coco_zalo', 'ctdet')
opt = opts().update_dataset_info_and_set_heads(opt, Dataset)
opt.vis_thresh = 0.7
#--flip_test --test_scales 0.5,0.75,1,1.25,1.5
opt.flip_test=True
opt.test_scales=[0.5,0.75,1,1.25,1.5]

print (opt)
#exit()
detector = detector_factory[opt.task](opt)

if False:
    # model forwarding
    image = cv2.imread(img_path)
    results = detector.run(image)['results']

    debug_image = visulize(image.copy(), results, opt.num_classes, opt.vis_thresh)

    plt.imshow(debug_image)
    plt.show()

if True:
    palette = [
        (0, 0, 255),
        (0, 255, 0),
        (255, 0, 0),
        (255, 255, 0),
        (0, 255, 255),
        (255, 0, 255),
        (127, 127, 0)
    ]
    paths = natsorted(glob.glob(os.path.join(test_data_dir, "*.png")))
    output_dir = "output_submission"
    final_output = []

    os.makedirs(output_dir, exist_ok=True)
    for path_index, path in enumerate(paths):
        print("%03d / %03d" % (path_index, len(paths)))
        image_id = int(os.path.splitext(os.path.basename(path))[0])

        image = cv2.imread(path)
        outputs = detector.run(image)['results']

        for j in range(1, opt.num_classes + 1):
            for bbox in outputs[j]:
                if bbox[4] < 0.2:
                    continue

                final_output.append({
                    "image_id": image_id,
                    "category_id": j,
                    "bbox": [float(bbox[0]), float(bbox[1]), float(bbox[2] - bbox[0]), float(bbox[3] - bbox[1])],
                    "score": float(bbox[4]),
                })

                cv2.rectangle(image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), palette[j-1], 2, cv2.LINE_AA)
                cv2.putText(image, lbl_id2name[j], (int(bbox[0]), int(bbox[1]) - 5), cv2.FONT_HERSHEY_TRIPLEX, 1,
                            (255, 0, 0), 1)

        cv2.imwrite(os.path.join(output_dir, os.path.basename(path)), image)

    json.dump(final_output, open(os.path.join(output_dir, "public_submission.json"), "w+"), indent=4)