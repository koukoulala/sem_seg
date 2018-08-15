import json
import argparse
import torch

from torch.autograd import Variable
from visualize.visualize import make_dot
from models.unet_upsample import Unet_upsample
from models.unet import Unet

def test(args):

    # Setup Data
    data_json = json.load(open('config.json'))
    x = Variable(torch.randn(32, 1, 128, 128))

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

    #visualize
    y=model(x)
    g=make_dot(y)
    g.view()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Hyperparams')
    parser.add_argument('--arch', nargs='?', type=str, default='unet',
                        help='Architecture to use [\' unet, unet_sample etc\']')
    parser.add_argument('--model', nargs='?', type=str, default='unet_best',
                        help='Path to the saved model')

    args = parser.parse_args()
    test(args)
