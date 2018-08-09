
import argparse
import pandas as pd

from loader import get_loader, get_data_path
from skimage.transform import resize
from utils import *

from sklearn.model_selection import train_test_split
from keras.preprocessing.image import load_img
from keras.models import load_model
from tqdm import tqdm_notebook
def test(args):

    # Setup Dataloader
    data_loader = get_loader(args.dataset)
    data_path = get_data_path(args.dataset)
    data = data_loader(data_path, img_size_ori=args.img_size_ori,img_size_target=args.img_size_target, img_norm=args.img_norm)

    train_df=data.train_df
    test_df=data.test_df
    img_size_ori=data.img_size_ori
    img_size_target=data.img_size_target

    def upsample(img):
        if img_size_ori == img_size_target:
            return img
        return resize(img, (img_size_target, img_size_target), mode='constant', preserve_range=True)

    def downsample(img):
        if img_size_ori == img_size_target:
            return img
        return resize(img, (img_size_ori, img_size_ori), mode='constant', preserve_range=True)

    #split data
    ids_train, ids_valid, x_train, x_valid, y_train, y_valid, cov_train, cov_test, depth_train, depth_test = train_test_split(
        train_df.index.values,
        np.array(train_df.images.map(upsample).tolist()).reshape(-1, img_size_target, img_size_target, 1),
        np.array(train_df.masks.map(upsample).tolist()).reshape(-1, img_size_target, img_size_target, 1),
        train_df.coverage.values,
        train_df.z.values,
        test_size=0.2, stratify=train_df.coverage_class, random_state=1337)

    # load Model and validate
    model = load_model(args.model_path)
    preds_valid = model.predict(x_valid).reshape(-1, img_size_target, img_size_target)
    preds_valid = np.array([downsample(x) for x in preds_valid])
    y_valid_ori = np.array([train_df.loc[idx].masks for idx in ids_valid])

    #score
    thresholds = np.linspace(0, 1, 50)
    ious = np.array(
        [iou_metric_batch(y_valid_ori, np.int32(preds_valid > threshold)) for threshold in tqdm_notebook(thresholds)])
    threshold_best_index = np.argmax(ious[9:-10]) + 9
    iou_best = ious[threshold_best_index]
    threshold_best = thresholds[threshold_best_index]
    print(iou_best,threshold_best)

    #Load, predict and submit the test image predictions.
    x_test = np.array(
        [upsample(np.array(load_img(data_path+"images/{}.png".format(idx), grayscale=True))) / 255 for idx in
         tqdm_notebook(test_df.index)]).reshape(-1, img_size_target, img_size_target, 1)
    preds_test = model.predict(x_test)
    pred_dict = {idx: RLenc(np.round(downsample(preds_test[i]) > threshold_best)) for i, idx in
                 enumerate(tqdm_notebook(test_df.index.values))}
    sub = pd.DataFrame.from_dict(pred_dict, orient='index')
    sub.index.names = ['id']
    sub.columns = ['rle_mask']
    sub.to_csv('./results/submission.csv')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Hyperparams')
    parser.add_argument('--model_path', nargs='?', type=str, default='./saved_models/unet_keras.model',
                        help='Path to the saved model')
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


    args = parser.parse_args()
    test(args)
