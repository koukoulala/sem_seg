import json
import argparse
import torch
from skimage.transform import resize

from torch.autograd import Variable
import pandas as pd
from utils import *

from loader.salt_loader import SaltLoader
from tqdm import tqdm_notebook
from torch.utils import data
from models.unet_upsample import Unet_upsample
from models.unet import Unet

def test(args):

    # Setup Dataloader
    data_json = json.load(open('config.json'))
    data_path = data_json[args.dataset]['data_path']

    t_loader = SaltLoader(data_path, split="test")
    test_df=t_loader.test_df
    test_loader = data.DataLoader(t_loader, batch_size=args.batch_size, num_workers=8)

    # load Model
    if args.arch=='unet':
        model = Unet(start_fm=16)
    else:
        model=Unet_upsample(start_fm=16)
    model_path = data_json[args.model]['model_path']
    model.load_state_dict(torch.load(model_path)['model_state'])
    model.cuda()
    total = sum([param.nelement() for param in model.parameters()])
    print('Number of params: %.2fM' % (total / 1e6))

    #test
    pred_list=[]
    for images in test_loader:
        images = Variable(images.cuda())
        y_preds = model(images)
        y_preds_shaped = y_preds.reshape(-1,  args.img_size_target, args.img_size_target)
        for idx in range(args.batch_size):
            y_pred = y_preds_shaped[idx]
            pred = torch.sigmoid(y_pred)
            pred = pred.cpu().data.numpy()
            pred_ori = resize(pred, (args.img_size_ori, args.img_size_ori), mode='constant', preserve_range=True)
            pred_list.append(pred_ori)

    #submit the test image predictions.
    threshold_best=args.threshold
    pred_dict = {idx: RLenc(np.round(pred_list[i] > threshold_best)) for i, idx in
                 enumerate(tqdm_notebook(test_df.index.values))}
    sub = pd.DataFrame.from_dict(pred_dict, orient='index')
    sub.index.names = ['id']
    sub.columns = ['rle_mask']
    sub.to_csv('./results/{}_submission.csv'.format(args.model))
    print("The submission.csv saved in ./results")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Hyperparams')
    parser.add_argument('--arch', nargs='?', type=str, default='unet_upsample',
                        help='Architecture to use [\' unet, unet_sample etc\']')
    parser.add_argument('--model', nargs='?', type=str, default='unet_upsample_best',
                        help='Path to the saved model')
    parser.add_argument('--dataset', nargs='?', type=str, default='salt',
                        help='Dataset to use [\' TGS etc\']')
    parser.add_argument('--img_size_ori', nargs='?', type=int, default=101,
                        help='Height of the input image')
    parser.add_argument('--img_size_target', nargs='?', type=int, default=128,
                        help='Height of the input image')

    parser.add_argument('--batch_size', nargs='?', type=int, default=32,
                        help='Batch Size')
    parser.add_argument('--threshold', nargs='?', type=float, default=0.5,
                        help='best threshold from validate')

    args = parser.parse_args()
    test(args)
