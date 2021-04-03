from configs.path_config import PathConfig
import numpy as np
import os
from glob import glob
from utils.read_and_write import read_json, write_json
path_config = PathConfig('scannet')

if __name__ == '__main__':
    scannet_train_split = 'datasets/splits/scannet/scannetv2_train.txt'
    scannet_val_split = 'datasets/splits/scannet/scannetv2_val.txt'

    fullscan_train_split = 'datasets/splits/fullscan/scannetv2_train.json'
    fullscan_val_split = 'datasets/splits/fullscan/scannetv2_val.json'

    if not os.path.exists(os.path.dirname(fullscan_train_split)):
        os.makedirs(os.path.dirname(fullscan_train_split))

    if not os.path.exists(os.path.dirname(fullscan_val_split)):
        os.makedirs(os.path.dirname(fullscan_val_split))

    scannet_train_list = list(np.loadtxt(scannet_train_split, dtype=str))
    scannet_val_list = list(np.loadtxt(scannet_val_split, dtype=str))

    fullscan_train_list = []
    fullscan_val_list = []


    '''read scan2cad annotation file'''
    scan2cad_annotation_path = path_config.scan2cad_annotation_path
    scan2cad_annotations = read_json(scan2cad_annotation_path)

    for annotation in scan2cad_annotations:
        scan_name = annotation['id_scan']
        bbox_file = os.path.join(path_config.processed_data_path, scan_name, 'bbox.pkl')
        if not os.path.exists(bbox_file):
            continue

        # save full scan files
        full_scan_file = os.path.join(path_config.processed_data_path, scan_name, 'full_scan.npz')
        assert os.path.exists(full_scan_file)
        data_items = {'scan': full_scan_file, 'bbox': bbox_file}
        if scan_name in scannet_train_list:
            fullscan_train_list.append(data_items)
        else:
            fullscan_val_list.append(data_items)

    write_json(fullscan_train_split, fullscan_train_list)
    write_json(fullscan_val_split, fullscan_val_list)