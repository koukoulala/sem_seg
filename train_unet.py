import sys, os
import torch
import argparse
import torch.nn as nn
from torch.utils import data

from torch.autograd import Variable

from models import get_model
from loader import get_loader, get_data_path
from skimage.transform import resize
import numpy as np

from sklearn.model_selection import train_test_split

def train(args):

    # Setup Dataloader
    data_loader = get_loader(args.dataset)
    data_path = get_data_path(args.dataset)
    data_all = data_loader(data_path, img_size_ori=args.img_size_ori,img_size_target=args.img_size_target, img_norm=args.img_norm)

    train_df=data_all.train_df
    img_size_ori=data_all.img_size_ori
    img_size_target=data_all.img_size_target

    def upsample(img):
        if img_size_ori == img_size_target:
            return img
        return resize(img, (img_size_target, img_size_target), mode='constant', preserve_range=True)

    #split data
    ids_train, ids_valid, x_train, x_valid, y_train, y_valid, cov_train, cov_test, depth_train, depth_test = train_test_split(
        train_df.index.values,
        np.array(train_df.images.map(upsample).tolist()).reshape(-1, img_size_target, img_size_target, 1),
        np.array(train_df.masks.map(upsample).tolist()).reshape(-1, img_size_target, img_size_target, 1),
        train_df.coverage.values,
        train_df.z.values,
        test_size=0.2, stratify=train_df.coverage_class, random_state=1337)
    x_train = np.append(x_train, [np.fliplr(x) for x in x_train], axis=0)
    y_train = np.append(y_train, [np.fliplr(x) for x in y_train], axis=0)
    print(x_train.shape)
    torch_dataset=data.TensorDataset(data_tensor=x_train,target_tensor=y_train)
    trainloader = data.DataLoader(torch_dataset, batch_size=args.batch_size, num_workers=8, shuffle=True)


    # Setup Model
    model = get_model(args.arch)
    model = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))
    model.cuda()

    # Check if model has custom optimizer / loss
    optimizer = torch.optim.SGD(model.parameters(), lr=args.l_rate, momentum=0.99, weight_decay=5e-4)
    loss_fn= nn.CrossEntropyLoss()

    best_loss=1000
    for epoch in range(args.n_epoch):
        model.train()
        for i, (images, labels) in enumerate(trainloader):
            images = Variable(images.cuda())
            labels = Variable(labels.cuda())

            optimizer.zero_grad()
            outputs = model(images)

            loss = loss_fn(input=outputs, target=labels)

            loss.backward()
            optimizer.step()

            if (i+1) % 20 == 0:
                print("Epoch [%d/%d] Loss: %.4f" % (epoch+1, args.n_epoch, loss.data[0]))
                if loss.data[0] >= best_loss:
                    best_loss = loss.data[0]
                    state = {'epoch': epoch + 1,
                             'model_state': model.state_dict(),
                             'optimizer_state': optimizer.state_dict(), }
                    torch.save(state, "./saved_models/{}_{}_best_model.pkl".format(args.arch, args.dataset))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Hyperparams')
    parser.add_argument('--arch', nargs='?', type=str, default='unet',
                        help='Architecture to use [\' unet, segnet etc\']')
    parser.add_argument('--dataset', nargs='?', type=str, default='TGS',
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

    parser.add_argument('--n_epoch', nargs='?', type=int, default=100,
                        help='# of the epochs')
    parser.add_argument('--batch_size', nargs='?', type=int, default=5,
                        help='Batch Size')
    parser.add_argument('--l_rate', nargs='?', type=float, default=1e-5, 
                        help='Learning Rate')

    args = parser.parse_args()
    train(args)
