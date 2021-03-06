import json
import torch
import argparse
import torch.nn as nn
import numpy as np
from torch.utils import data

from torch.autograd import Variable

from models.unet import Unet
from models.unet_upsample import Unet_upsample
from loader.salt_loader import SaltLoader

def train(args):

    # Setup Dataloader
    data_json = json.load(open('config.json'))
    data_path=data_json[args.dataset]['data_path']
    t_loader = SaltLoader(data_path, img_size_ori=args.img_size_ori,img_size_target=args.img_size_target)
    v_loader = SaltLoader(data_path, split='val',img_size_ori=args.img_size_ori, img_size_target=args.img_size_target)

    train_loader = data.DataLoader(t_loader, batch_size=args.batch_size, num_workers=8, shuffle=True)
    val_loader = data.DataLoader(v_loader, batch_size=args.batch_size, num_workers=8)

    # Setup Model
    if args.arch=='unet':
        model = Unet(start_fm=16)
    else:
        model=Unet_upsample(start_fm=16)
    print(model)
    total = sum([param.nelement() for param in model.parameters()])
    print('Number of params: %.2fM' % (total / 1e6))

    model.cuda()

    # Check if model has custom optimizer / loss
    optimizer = torch.optim.Adam(model.parameters(), lr=args.l_rate)
    loss_fn= nn.BCEWithLogitsLoss()

    best_loss=100
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
        if np.mean(val_losses) < best_loss:
            best_loss = np.mean(val_losses)
            state = {'epoch': epoch + 1,
                     'model_state': model.state_dict(),
                     'optimizer_state': optimizer.state_dict(), }
            torch.save(state, "./saved_models/{}_{}_best_model.pkl".format(args.arch, args.dataset))

        # Print Loss
        print('Epoch: {}. Train Loss: {}. Val Loss: {}'.format(epoch + 1, np.mean(train_losses), np.mean(val_losses)))

    state = {'model_state': model.state_dict(),
             'optimizer_state': optimizer.state_dict(), }
    torch.save(state, "./saved_models/{}_{}_final_model.pkl".format(args.arch, args.dataset))

    print("saved two models in ./saved_models")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Hyperparams')
    parser.add_argument('--arch', nargs='?', type=str, default='unet_upsample',
                        help='Architecture to use [\' unet, unet_sample etc\']')
    parser.add_argument('--dataset', nargs='?', type=str, default='salt',
                        help='Dataset to use [\' salt etc\']')
    parser.add_argument('--img_size_ori', nargs='?', type=int, default=101,
                        help='Height of the input image')
    parser.add_argument('--img_size_target', nargs='?', type=int, default=128,
                        help='Height of the input image')

    parser.add_argument('--n_epoch', nargs='?', type=int, default=70,
                        help='# of the epochs')
    parser.add_argument('--batch_size', nargs='?', type=int, default=32,
                        help='Batch Size')
    parser.add_argument('--l_rate', nargs='?', type=float, default=1e-3,
                        help='Learning Rate')

    args = parser.parse_args()
    train(args)
