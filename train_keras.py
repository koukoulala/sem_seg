
import argparse

from models import get_model
from loader import get_loader, get_data_path
from skimage.transform import resize
import numpy as np

from sklearn.model_selection import train_test_split
from keras import Model
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
def train(args):

    # Setup Dataloader
    data_loader = get_loader(args.dataset)
    data_path = get_data_path(args.dataset)
    data = data_loader(data_path, img_size_ori=args.img_size_ori,img_size_target=args.img_size_target, img_norm=args.img_norm)

    train_df=data.train_df
    img_size_ori=data.img_size_ori
    img_size_target=data.img_size_target

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
    #print(x_train.shape,y_train.shape)

    # Setup Model
    model_build = get_model(args.arch)
    model = Model(model_build.input_layer, model_build.output_layer)
    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
    model.summary()

    #training
    early_stopping = EarlyStopping(patience=10, verbose=1)
    model_checkpoint = ModelCheckpoint("./saved_models/unet_keras.model", save_best_only=True, verbose=1)
    reduce_lr = ReduceLROnPlateau(factor=0.1, patience=5, min_lr=0.00001, verbose=1)

    epochs = args.n_epoch
    batch_size = args.batch_size


    model.fit(x_train, y_train,
              validation_data=[x_valid, y_valid],
              epochs=epochs,
              batch_size=batch_size,
              callbacks=[early_stopping, model_checkpoint, reduce_lr])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Hyperparams')
    parser.add_argument('--arch', nargs='?', type=str, default='unet_keras',
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

    parser.add_argument('--n_epoch', nargs='?', type=int, default=15,
                        help='# of the epochs')
    parser.add_argument('--batch_size', nargs='?', type=int, default=32,
                        help='Batch Size')
    parser.add_argument('--l_rate', nargs='?', type=float, default=1e-5, 
                        help='Learning Rate')

    args = parser.parse_args()
    train(args)
