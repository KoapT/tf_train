import json
import os

infile = '/home/psdz/TK/Datasets/pascalvoc/label_list'
outfile = 'label_map.json'

names = []
with open(infile, 'r') as f:
    for line in f.readlines():
        text = line.strip()
        if text != 'background':
            names.append(text)
label_list = []
for i, name in enumerate(names):
    label_list.append(dict(id=i, name=name))
    label_json = json.dumps(label_list).replace(',', ',\n')
print(label_json)
with open(outfile, 'w') as w:
    w.write(label_json)
