from models.unet import *
from models.unet_keras import *


def get_model(name, n_classes=0):
    model = _get_model_instance(name)

    if name== 'unet_keras':
        model = model(
            start_neurons=16,
            img_size_target=128
        )
    elif name== 'unet':
        model=model(
        start_filters=128,
        in_channels=1
    )
    else:
        model = model(n_classes=n_classes)

    return model

def _get_model_instance(name):
    try:
        return {
            'unet': unet,
            'unet_keras':unet_keras,
        }[name]
    except:
        print('Model {} not available'.format(name))
