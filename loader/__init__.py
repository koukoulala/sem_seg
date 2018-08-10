import json

from loader.TGS_loader import TGSLoader
from loader.salt_loader import SaltLoader


def get_loader(name):
    """get_loader

    :param name:
    """
    return {
        'TGS': TGSLoader,
        'salt':SaltLoader,
    }[name]


def get_data_path(name, config_file='config.json'):
    """get_data_path

    :param name:
    :param config_file:
    """
    data = json.load(open(config_file))
    return data[name]['data_path']
