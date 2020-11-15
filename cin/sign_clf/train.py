import numpy as np
import random
import os
from natsort import natsorted
import glob
from copy import deepcopy
import torch
from torch.utils.data import DataLoader
from torch import optim
from torch.optim import lr_scheduler
import torch.nn as nn
from torchvision.transforms import transforms

from libs.net import SignNet
from libs.sign_dataset import SignDataset

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def save_best(net, optimzier, epoch, output_dir, params):
    d = {
        'net': deepcopy(net).state_dict(),
        'optimizer': deepcopy(optimzier).state_dict(),
        'params': params
    }
    path = os.path.join(output_dir, "model_%03d.pth" % epoch)
    torch.save(d, path)

    print ('save the best in %s ...' % path)

def load_best(net, optimizer, input_dir):
    weight_paths = natsorted(glob.glob(os.path.join(input_dir, "*.pth")))

    if len(weight_paths) == 0:
        return 0

    weight_path = weight_paths[-1]
    weight_data = torch.load(weight_path)

    net.load_state_dict(weight_data["net"])
    optimizer.load_state_dict(weight_data["optimizer"])

    epoch = int(os.path.basename(weight_path)[6:9]) + 1
    return epoch

def calc_accuracy(output, label):
    n_sample = output.size()[0]

    predict = torch.argmax(output, dim=-1)
    correct = torch.sum(predict == label, dtype=torch.float)

    accuracy = correct / (1e-6 + n_sample)
    return accuracy

def gen_transform(input_size):
    data_transforms = {
        'train': transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(input_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),

        'val': transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(input_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    return data_transforms

global_iter = 0
def main(params, image_dir, train_annot_path, valid_annot_path):
    p = params
    device = p['device']
    data_transform = gen_transform(p['input_size'])

    train_dataset = SignDataset(image_dir, train_annot_path, data_transform['train'], p['lower_h'], p['lower_w'])
    train_loader  = DataLoader(train_dataset, batch_size=p['batch_size'], shuffle=True, num_workers=p['n_worker'])

    train_label_freqs = train_dataset.get_label_freq()
    train_label_weights = np.max(train_label_freqs).astype(np.float32) / train_label_freqs

    print ('train n_dataset:', len(train_dataset))
    print ('tran lbl frequencies:', train_label_freqs)
    print ('train lbl weights:', train_label_weights)

    n_class = len(train_dataset.annot_names)
    net = SignNet(num_classes=n_class)

    optimizer = optim.Adam(net.parameters(), lr=p['base_lr'])
    criterion = nn.CrossEntropyLoss(weight=torch.from_numpy(train_label_weights).float())

    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=p['step_size'], gamma=0.1)

    loaded_epoch = load_best(net, optimizer, p['weight_dir'])

    net.to(device)
    criterion.to(device)

    best_loss = np.inf
    for e_i in range(loaded_epoch, p['max_epoch']):
        loss = train(train_loader, net, criterion, optimizer, p)
        exp_lr_scheduler.step(epoch=e_i)

        if loss < best_loss:
            best_loss = loss

            # TODO: calc validation accuracy/loss here

            save_best(net, optimizer, e_i, p['weight_dir'], params)

def train(train_loader, net, criterion, optimizer, params):
    global global_iter

    e_loss = []
    for batch_id, batch_data in enumerate(train_loader):
        net.train()
        images, labels = batch_data

        images = images.to(params['device'])
        labels = labels.to(params['device'])

        output = net(images)
        loss = criterion(output, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        global_iter += 1
        if global_iter % params['print_freq'] == 0:
            accuracy = calc_accuracy(output, labels)
            print ('iter: %05d, loss: %.5f, accuracy: %.5f ...' % (global_iter, loss.item(), accuracy.cpu().numpy()))

        e_loss += [loss.item()]

    return np.mean(e_loss)

if __name__ == '__main__':
    set_seed(2312)

    global_params = {
        'input_size': (224,224),
        'n_class': 7,
        'lower_h': 30,
        'lower_w': 30,

        'max_epoch': 25,
        'batch_size': 32,
        'n_worker': 4,

        'base_lr': 0.001,
        'step_size': 2,

        'device': 'cuda',
        'print_freq': 10,
        'weight_dir': 'weights'
    }

    image_dir = "/home/kan/Desktop/cinnamon/zalo/CenterNet/data/coco_zalo/images"
    train_annot_path = "/home/kan/Desktop/cinnamon/zalo/CenterNet/data/coco_zalo/train.json"
    valid_annot_path = ""

    global_params['device'] = torch.device(global_params['device'] if torch.cuda.is_available() else 'cpu')
    os.makedirs(global_params['weight_dir'], exist_ok=True)
    main(global_params, image_dir, train_annot_path, valid_annot_path)

