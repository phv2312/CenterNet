import os
import cv2
import numpy as np
from PIL import Image

from torch.utils.data import Dataset, DataLoader
from pycocotools.coco import COCO

def parse_from_coco(annot_path, lower_h, lower_w):
    #
    pair = []
    coco_dataset = COCO(annot_path)

    #
    categories = coco_dataset.loadCats(coco_dataset.getCatIds())
    category_names = [category['name'] for category in categories]

    #
    image_ids = coco_dataset.getImgIds(catIds=[])
    for image_id in image_ids:
        image_data  = coco_dataset.loadImgs(image_id)[0]
        annot_id    = coco_dataset.getAnnIds(imgIds=image_data['id'], catIds=[], iscrowd=None)
        annot_datas = coco_dataset.loadAnns(annot_id)

        for annot_data in annot_datas:
            x_min, y_min, w, h = annot_data['bbox']

            if w > lower_w and h > lower_h:

                pair += [(image_data['file_name'], (x_min, y_min, x_min + w, y_min + h),annot_data['category_id'])]

    return pair, category_names

class SignDataset(Dataset):
    def __init__(self, im_dir, annot_path, transform=None, lower_h=40, lower_w=40):
        super(SignDataset, self).__init__()

        self.im_dir = im_dir
        self.annot_path = annot_path
        self.pair, self.annot_names = parse_from_coco(annot_path, lower_h, lower_w)

        self.transform = transform

    def get_label_freq(self):
        label_ids = np.array([p[-1] for p in self.pair])
        label_unique_ids, label_count = np.unique(label_ids, return_counts=True)

        perm = np.argsort(label_unique_ids)
        return label_count[perm]

    def lblid_to_name(self, lblid):
        return self.annot_names[lblid]

    def __len__(self):
        return len(self.pair)

    def __getitem__(self, idx):
        cur_pair = self.pair[idx]
        im_name, bbox, lbl_id = cur_pair
        x_min, y_min, x_max, y_max = bbox

        im = cv2.imread(os.path.join(self.im_dir, im_name))
        im = im[y_min:y_max, x_min:x_max, :]

        if self.transform is not None:
            im = self.transform(im)

        return im, lbl_id - 1

if __name__ == '__main__':
    from torchvision.transforms import transforms
    import matplotlib.pyplot as plt

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(size=(224, 224)),
        transforms.ToTensor()
    ])

    # test
    im_dir = "./../../../data/coco_zalo/images"
    annot_path = "./../../../data/coco_zalo/train.json"

    train_dataset   = SignDataset(im_dir, annot_path, transform, lower_h=30, lower_w=30)
    print ('n_dataset:', len(train_dataset))
    print ('label frequency:', train_dataset.get_label_freq())
    train_loader    = DataLoader(train_dataset, batch_size=1, shuffle=False, num_workers=0)
    train_iter      = iter(train_loader)

    for batch_id, batch_data in enumerate(train_iter):
        images, labels = batch_data

        labels = labels.detach().cpu()[0].numpy()

        images = (images + 1) / 2
        images = images.detach().cpu()[0].permute(1,2,0).numpy()
        images = (images * 255).astype(np.uint8)

        print ('>>> id:', batch_id, ', label:', labels, ', label_name:', train_dataset.lblid_to_name(labels))

        plt.imshow(images)
        plt.show()
