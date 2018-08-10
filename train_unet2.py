import sys, os
import torch
import argparse
import torch.nn as nn
import numpy as np
from torch.utils import data

from torch.autograd import Variable

from models import get_model
from loader import get_loader, get_data_path

def train(args):

    # Setup Dataloader
    data_loader = get_loader(args.dataset)
    data_path = get_data_path(args.dataset)
    t_loader = data_loader(data_path, img_size_ori=args.img_size_ori,img_size_target=args.img_size_target, img_norm=args.img_norm)
    v_loader = data_loader(data_path, split='val',img_size_ori=args.img_size_ori, img_size_target=args.img_size_target,
                           img_norm=args.img_norm)

    train_loader = data.DataLoader(t_loader, batch_size=args.batch_size, num_workers=8, shuffle=True)
    val_loader = data.DataLoader(v_loader, batch_size=args.batch_size, num_workers=8)

    # Setup Model
    model = get_model(args.arch)
    print(model)
    #model = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))
    model.cuda()

    # Check if model has custom optimizer / loss
    optimizer = torch.optim.Adam(model.parameters(), lr=args.l_rate)
    loss_fn= nn.BCEWithLogitsLoss()

    mean_train_losses = []
    mean_val_losses = []
    for epoch in range(args.n_epoch):
        train_losses = []
        val_losses = []
        for images, masks in train_loader:
            images = Variable(images.cuda())
            masks = Variable(masks.cuda())

            outputs = model(images)

            loss = loss_fn(outputs, masks)
            train_losses.append(loss.data)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        for images, masks in val_loader:
            images = Variable(images.cuda())
            masks = Variable(masks.cuda())

            outputs = model(images)
            loss = loss_fn(outputs, masks)
            val_losses.append(loss.data)

        mean_train_losses.append(np.mean(train_losses))
        mean_val_losses.append(np.mean(val_losses))
        # Print Loss
        print('Epoch: {}. Train Loss: {}. Val Loss: {}'.format(epoch + 1, np.mean(train_losses), np.mean(val_losses)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Hyperparams')
    parser.add_argument('--arch', nargs='?', type=str, default='unet2',
                        help='Architecture to use [\' unet, segnet etc\']')
    parser.add_argument('--dataset', nargs='?', type=str, default='salt',
                        help='Dataset to use [\' TGS etc\']')
    parser.add_argument('--img_size_ori', nargs='?', type=int, default=101,
                        help='Height of the input image')
    parser.add_argument('--img_size_target', nargs='?', type=int, default=128,
                        help='Height of the input image')

    parser.add_argument('--img_norm', dest='img_norm', action='store_true', 
                        help='Enable input image scales normalization [0, 1] | True by default')
    parser.add_argument('--no-img_norm', dest='img_norm', action='store_false', 
                        help='Disable input image scales normalization [0, 1] | True by default')
    parser.set_defaults(img_norm=False)

    parser.add_argument('--n_epoch', nargs='?', type=int, default=10,
                        help='# of the epochs')
    parser.add_argument('--batch_size', nargs='?', type=int, default=1,
                        help='Batch Size')
    parser.add_argument('--l_rate', nargs='?', type=float, default=1e-5, 
                        help='Learning Rate')

    args = parser.parse_args()
    train(args)