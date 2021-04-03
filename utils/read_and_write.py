import numpy as np
import re
import json
import os

def read_obj(model_path, flags = ('v')):
    fid = open(model_path, 'r', encoding="utf-8")

    data = {}

    for head in flags:
        data[head] = []

    for line in fid:
        line = line.strip()
        if not line:
            continue
        line = re.split('\s+', line)
        if line[0] in flags:
            data[line[0]].append(line[1:])

    fid.close()

    if 'v' in data.keys():
        data['v'] = np.array(data['v']).astype(np.float)

    if 'vt' in data.keys():
        data['vt'] = np.array(data['vt']).astype(np.float)

    if 'vn' in data.keys():
        data['vn'] = np.array(data['vn']).astype(np.float)

    return data

def read_txt(txt_file_list):
    '''
    read txt files and output a matrix.
    :param exr_file_list:
    :return:
    '''
    if isinstance(txt_file_list, str):
        txt_file_list = [txt_file_list]

    output_list = []
    for txt_file in txt_file_list:
        output_list.append(np.loadtxt(txt_file))

    return np.array(output_list)

def write_obj(objfile, data):
    '''
    Write data into obj_file.
    :param objfile (str): file path.
    :param data (dict): obj contents to be writen.
    :return:
    '''
    with open(objfile, 'w+') as file:
        for key, item in data.items():
            for point in item:
                file.write(key + ' %s' * len(point) % tuple(point) + '\n')

def read_json(file):
    '''
    read json file
    :param file: file path.
    :return:
    '''
    with open(file, 'r') as f:
        output = json.load(f)
    return output

def write_json(file, data):
    '''
    read json file
    :param file: file path.
    :param data: dict content
    :return:
    '''
    assert os.path.exists(os.path.dirname(file))

    with open(file, 'w') as f:
        json.dump(data, f)