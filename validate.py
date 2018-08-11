import json
import argparse
import torch
from skimage.transform import resize
from sklearn.metrics import jaccard_similarity_score

from torch.autograd import Variable
import pandas as pd
from utils import *

from loader.salt_loader import SaltLoader
from tqdm import tqdm_notebook
from torch.utils import data
from models.unet import Unet

def test(args):

    # Setup Dataloader
    data_json = json.load(open('config.json'))
    data_path = data_json[args.dataset]['data_path']
    v_loader = SaltLoader(data_path, split='val', split_size=0.5)
    train_df=v_loader.train_df

    val_loader = data.DataLoader(v_loader, batch_size=args.batch_size, num_workers=8)

    # load Model
    model = Unet()
    model.load_state_dict(torch.load(args.model_path)['model_state'])
    model.cuda()
    model.eval()

    #validate
    pred_list=[]
    for images, masks in val_loader:
        images = Variable(images.cuda())
        y_preds = model(images)
        # print(y_preds.shape)
        y_preds_shaped = y_preds.reshape(-1, 128, 128)
        for idx in range(args.batch_size):
            y_pred = y_preds_shaped[idx]
            pred = torch.sigmoid(y_pred)
            pred = pred.cpu().data.numpy()
            pred_ori = resize(pred, (101, 101), mode='constant', preserve_range=True)
            pred_list.append(pred_ori)


    preds_valid = np.array(pred_list)
    y_valid_ori = np.array([train_df.loc[idx].masks for idx in v_loader.ids_valid])

    #jaccard
    for threshold in np.linspace(0, 1, 11):
        ious = []
        for y_pred, mask in zip(preds_valid, y_valid_ori):
            prediction = (y_pred > threshold).astype(int)
            iou = jaccard_similarity_score(mask.flatten(), prediction.flatten())
            ious.append(iou)

        accuracies = [np.mean(ious > iou_threshold)
                      for iou_threshold in np.linspace(0.5, 0.95, 10)]
        print('Threshold: %.1f, Metric: %.3f' % (threshold, np.mean(accuracies)))

    #score
    thresholds = np.linspace(0, 1, 50)
    ious = np.array(
        [iou_metric_batch(y_valid_ori, np.int32(preds_valid > threshold)) for threshold in tqdm_notebook(thresholds)])
    threshold_best_index = np.argmax(ious[9:-10]) + 9
    iou_best = ious[threshold_best_index]
    threshold_best = thresholds[threshold_best_index]
    print(iou_best,threshold_best)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Hyperparams')
    parser.add_argument('--model_path', nargs='?', type=str, default='./saved_models/unet_salt_best_model.pkl',
                        help='Path to the saved model')
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


    args = parser.parse_args()
    test(args)
