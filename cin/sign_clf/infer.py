import torch

from libs.net import SignNet
from libs.sign_dataset import SignDataset
from train import gen_transform

class Inference():
    def __init__(self, model_weight_path, device):
        self.device = device

        weight_data = torch.load(model_weight_path, map_location='cpu')
        self.params = weight_data['params']
        self.base_model = SignNet(num_classes=self.params['n_class'])
        self.base_model.load_state_dict(state_dict=weight_data['net'])
        self.base_model.to(self.device)
        self.base_model.eval()

        #
        self.val_transform = gen_transform(input_size=self.params['input_size'])['val']
        print ('finished load model from %s ...' % model_weight_path)

    def infer(self, np_image):
        tensor_input = self.val_transform(np_image)
        tensor_input = tensor_input.unsqueeze(0)
        tensor_input = tensor_input.to(self.device)

        with torch.no_grad():
            output = self.base_model(tensor_input)

        predict_id = torch.argmax(output, dim=-1)
        predict_id = predict_id.cpu().numpy()

        return predict_id

if __name__ == '__main__':
    #
    from torch.utils.data import DataLoader
    import matplotlib.pyplot as plt

    #
    def imgshow(im):
        plt.imshow(im)
        plt.show()

    #
    device = torch.device('cuda')

    #
    model_weight_path = "./weights/model_005.pth"
    model = Inference(model_weight_path, device)

    # test
    im_dir = "./../../data/coco_zalo/images"
    annot_path = "./../../data/coco_zalo/train.json"

    val_dataset = SignDataset(im_dir, annot_path, lower_h=30, lower_w=30)
    val_loader  = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=1)
    val_iter    = iter(val_loader)

    for batch_id, batch_data in enumerate(val_iter):
        images, labels = batch_data

        images = images.detach().cpu().numpy()[0] # from tensor to numpy, (h,w,3)
        labels = labels.detach().cpu().numpy()[0]

        predict_id = model.infer(images)[0]
        predict_name = val_dataset.lblid_to_name(predict_id)
        gt_name = val_dataset.lblid_to_name(labels)

        print ('predict: %s,  gt: %s ...' % (predict_name, gt_name))
        imgshow(images)


