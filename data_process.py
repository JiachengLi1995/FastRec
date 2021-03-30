import json
from collections import defaultdict
import gzip
import random
from tqdm import tqdm
import argparse
import os

class LabelField:
    def __init__(self):
        self.label2id = dict()
        self.label_num = 0

    def get_id(self, label):
        
        if label in self.label2id:
            return self.label2id[label]
        
        self.label2id[label] = self.label_num
        self.label_num += 1

        return self.label2id[label]


parser = argparse.ArgumentParser()
    
parser.add_argument('--file_path', default='Beauty.json.gz', help='Processing file path (.gz file).')
parser.add_argument('--output_path', default='Beauty', help='Output directory')
args = parser.parse_args()

output_path = args.output_path
if not os.path.exists(output_path):
    os.mkdir(output_path)


input_file = args.file_path
train_file = os.path.join(output_path, 'train.json')
dev_file = os.path.join(output_path, 'val.json')
test_file = os.path.join(output_path, 'test.json')
umap_file = os.path.join(output_path, 'umap.json')
smap_file = os.path.join(output_path, 'smap.json')



user_field = LabelField()
s_field = LabelField()
sequences = defaultdict(list)

gin = gzip.open(input_file, 'rb')

for line in tqdm(gin):

    line = json.loads(line)

    user_id = line['reviewerID']
    item_id = line['asin']
    time = line['unixReviewTime']
    
    sequences[user_field.get_id(user_id)].append((s_field.get_id(item_id), time))

train_dict = dict()
dev_dict = dict()
test_dict = dict()

for k, v in tqdm(sequences.items()):
    sequences[k] = sorted(v, key=lambda x: x[1])
    sequences[k] = [ele[0] for ele in sequences[k]]

    length = len(sequences[k])
    if length<3:
        train_dict[k] = sequences[k]
    else:
        train_dict[k] = sequences[k][:length-2]
        dev_dict[k] = [sequences[k][length-2]]
        test_dict[k] = [sequences[k][length-1]]


f_u = open(umap_file, 'w', encoding='utf8')
json.dump(user_field.label2id, f_u)
f_u.close()

f_s = open(smap_file, 'w', encoding='utf8')
json.dump(s_field.label2id, f_s)
f_s.close()

train_f = open(train_file, 'w', encoding='utf8')
json.dump(train_dict, train_f)
train_f.close()

dev_f = open(dev_file, 'w', encoding='utf8')
json.dump(dev_dict, dev_f)
dev_f.close()

test_f = open(test_file, 'w', encoding='utf8')
json.dump(test_dict, test_f)
test_f.close()



