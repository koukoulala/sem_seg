import json
import torch
import argparse
import torch.nn as nn
import numpy as np
from sklearn.metrics import jaccard_similarity_score
from torch.utils import data

from torch.autograd import Variable

from models.unet import Unet
from loader.salt_loader import SaltLoader

def train(args):

    # Setup Dataloader
    data_json = json.load(open('config.json'))
    data_path=data_json[args.dataset]['data_path']
    t_loader = SaltLoader(data_path, img_size_ori=args.img_size_ori,img_size_target=args.img_size_target, img_norm=args.img_norm)
    v_loader = SaltLoader(data_path, split='val',img_size_ori=args.img_size_ori, img_size_target=args.img_size_target,
                           img_norm=args.img_norm)

    train_loader = data.DataLoader(t_loader, batch_size=args.batch_size, num_workers=8, shuffle=True)
    val_loader = data.DataLoader(v_loader, batch_size=args.batch_size, num_workers=8)

    # Setup Model
    model = Unet()
    print(model)

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

            if loss<best_loss:
                best_loss=loss
                state = {'epoch': epoch + 1,
                         'model_state': model.state_dict(),
                         'optimizer_state': optimizer.state_dict(), }
                torch.save(state, "./saved_models/{}_{}_best_model.pkl".format(args.arch, args.dataset))

        mean_train_losses.append(np.mean(train_losses))
        mean_val_losses.append(np.mean(val_losses))
        # Print Loss
        print('Epoch: {}. Train Loss: {}. Val Loss: {}'.format(epoch + 1, np.mean(train_losses), np.mean(val_losses)))

    print("saved model in ./saved_models/{}_{}_best_model.pkl".format(args.arch, args.dataset))

    y_pred_true_pairs = []
    for images, masks in val_loader:
        images = Variable(images.cuda())
        y_preds = model(images)
        for i, _ in enumerate(images):
            y_pred = y_preds[i]
            y_pred = torch.sigmoid(y_pred)
            y_pred = y_pred.cpu().data.numpy()
            y_pred_true_pairs.append((y_pred, masks[i].numpy()))

    # https://www.kaggle.com/leighplt/goto-pytorch-fix-for-v0-3
    for threshold in np.linspace(0, 1, 11):

        ious = []
        for y_pred, mask in y_pred_true_pairs:
            prediction = (y_pred > threshold).astype(int)
            iou = jaccard_similarity_score(mask.flatten(), prediction.flatten())
            ious.append(iou)

        accuracies = [np.mean(ious > iou_threshold)
                      for iou_threshold in np.linspace(0.5, 0.95, 10)]
        print('Threshold: %.1f, Metric: %.3f' % (threshold, np.mean(accuracies)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Hyperparams')
    parser.add_argument('--arch', nargs='?', type=str, default='unet',
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
    parser.add_argument('--batch_size', nargs='?', type=int, default=16,
                        help='Batch Size')
    parser.add_argument('--l_rate', nargs='?', type=float, default=1e-3,
                        help='Learning Rate')

    args = parser.parse_args()
    train(args)
