# Utility functions in training and testing
# author: ynie
# date: Feb, 2020

import os
import yaml
import logging
from datetime import datetime


def update_recursive(dict1, dict2):
    ''' Update two config dictionaries recursively.

    Args:
        dict1 (dict): first dictionary to be updated
        dict2 (dict): second dictionary which entries should be used

    '''
    for k, v in dict2.items():
        if k not in dict1:
            dict1[k] = dict()
        if isinstance(v, dict):
            update_recursive(dict1[k], v)
        else:
            dict1[k] = v

class CONFIG(object):
    '''
    Stores all configures
    '''
    def __init__(self, input=None):
        '''
        Loads config file
        :param path (str): path to config file
        :return:
        '''
        self.config = self.read_to_dict(input)
        self._logger, self._save_path = self.load_logger()

        # update save_path to config file
        self.update_config(log={'path': self._save_path})

        # update visualization path
        vis_path = os.path.join(self._save_path, self.config['log']['vis_path'])
        if not os.path.exists(vis_path):
            os.mkdir(vis_path)
        self.update_config(log={'vis_path': vis_path})

        # initiate device environments
        os.environ["CUDA_VISIBLE_DEVICES"] = self.config['device']['gpu_ids']

    @property
    def logger(self):
        return self._logger

    @property
    def save_path(self):
        return self._save_path

    def load_logger(self):
        # set file handler
        save_path = os.path.join(self.config['log']['path'], datetime.now().isoformat())
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        logfile = os.path.join(save_path, 'log.txt')
        file_handler = logging.FileHandler(logfile)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        self.__file_handler = file_handler

        # configure logger
        logger = logging.getLogger('Empty')
        logger.setLevel(logging.INFO)
        logger.addHandler(file_handler)

        return logger, save_path

    def log_string(self, content):
        self._logger.info(content)
        print(content)

    def read_to_dict(self, input):
        if not input:
            return dict()
        if isinstance(input, str) and os.path.isfile(input):
            if input.endswith('yaml'):
                with open(input, 'r') as f:
                    config = yaml.load(f, Loader=yaml.FullLoader)
            else:
                ValueError('Config file should be with the format of *.yaml')
        elif isinstance(input, dict):
            config = input
        else:
            raise ValueError('Unrecognized input type (i.e. not *.yaml file nor dict).')

        return config

    def update_config(self, *args, **kwargs):
        '''
        update config and corresponding logger setting
        :param input: dict settings add to config file
        :return:
        '''
        cfg1 = dict()
        for item in args:
            cfg1.update(self.read_to_dict(item))

        cfg2 = self.read_to_dict(kwargs)

        new_cfg = {**cfg1, **cfg2}

        update_recursive(self.config, new_cfg)
        # when update config file, the corresponding logger should also be updated.
        self.__update_logger()

    def write_config(self):
        output_file = os.path.join(self._save_path, 'out_config.yaml')

        with open(output_file, 'w') as file:
            yaml.dump(self.config, file, default_flow_style = False)

    def __update_logger(self):
        # configure logger
        name = self.config['mode'] if 'mode' in self.config else self._logger.name
        logger = logging.getLogger(name)
        logger.setLevel(logging.INFO)
        logger.addHandler(self.__file_handler)
        self._logger = logger

def mount_external_config(cfg):
    if cfg.config['data']['dataset'] == 'scannet':
        from configs.scannet_config import ScannetConfig
        dataset_config = ScannetConfig()
        setattr(cfg, 'dataset_config', dataset_config)

        # Used for AP calculation
        eval_cfg = cfg.config.get('val', cfg.config.get('test'))
        CONFIG_DICT = {'remove_empty_box': not eval_cfg['faster_eval'],
                       'use_3d_nms': eval_cfg['use_3d_nms'],
                       'nms_iou': eval_cfg['nms_iou'],
                       'use_old_type_nms': eval_cfg['use_old_type_nms'],
                       'cls_nms': eval_cfg['use_cls_nms'],
                       'per_class_proposal': eval_cfg['per_class_proposal'],
                       'conf_thresh': eval_cfg['conf_thresh'],
                       'dataset_config': dataset_config}

        setattr(cfg, 'eval_config', CONFIG_DICT)
    return cfg