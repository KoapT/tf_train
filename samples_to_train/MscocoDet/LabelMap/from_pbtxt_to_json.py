import json
import os

infile = 'label_map.pbtxt'
outfile = os.path.splitext(infile)[0] + '.json'

names = []
with open(infile, 'r') as f:
    for line in f.readlines():
        text = line.strip()
        if 'name' in text:
            names.append(text.split(':')[-1].strip().strip("\'"))
label_list = []
for i, name in enumerate(names):
    label_list.append(dict(id=str(i + 1), name=name))
    label_json = json.dumps(label_list).replace(',', ',\n')
print(label_json)
with open(outfile, 'w') as w:
    w.write(label_json)
