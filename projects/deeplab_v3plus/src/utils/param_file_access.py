import json
import numpy as np





def get_json_params(file_path):
    params = []
    with open(file_path, 'r') as f:
        params = json.load(f)
    return params

def get_txt_params(file_path, splitStr=' '):
    params = []
    with open(file_path,'r') as f:
        for line in f:
            params.append(list(line.strip().split(splitStr)))
    return params

def get_label_map_palette(label_map_path):
    labels = get_json_params(label_map_path)
    palette = np.array([[0, 0, 0] for i in range(256)]).astype(np.uint8)
    for label in labels:
        palette[label['id'], 0] = label['vis']['r']
        palette[label['id'], 1] = label['vis']['g']
        palette[label['id'], 2] = label['vis']['b']
    return palette