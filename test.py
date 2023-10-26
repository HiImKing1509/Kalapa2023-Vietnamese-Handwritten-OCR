import torch
from torch.autograd import Variable
from models.utils import strLabelConverter, resizePadding
from PIL import Image
import sys
import models.crnn as crnn
import argparse
from torch.nn.functional import softmax
import numpy as np
import time
import os
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument('--path', required=True, help='path to test')
parser.add_argument('--alphabet', required=True, help='path to vocab')
parser.add_argument('--model', required=True, help='path to model')
parser.add_argument('--imgW', type=int, default=None, help='path to model')
parser.add_argument('--imgH', type=int, default=32, help='path to model')

opt = parser.parse_args()


def custom_sort(filename):
    numeric_part = "".join(filter(str.isdigit, filename))
    return int(numeric_part)


def numerical_sort(filename):
    if 'DS_Store' not in filename:
        num = int(filename.split('.')[0])
        return num
    else:
        pass


alphabet = open(opt.alphabet).read().rstrip()
nclass = len(alphabet) + 1
nc = 3

model = crnn.CRNN(opt.imgH, nc, nclass, 256)
if torch.cuda.is_available():
    model = model.cuda()
model.eval()

print('loading pretrained model from %s' % opt.model)
model.load_state_dict(torch.load(opt.model, map_location='cpu'))
converter = strLabelConverter(alphabet, ignore_case=False)

data = pd.DataFrame(columns=['id', 'answer'])

for subfolder in sorted(os.listdir(opt.path), key=custom_sort):
    subfolder_path = os.path.join(opt.path, subfolder)
    if os.path.isdir(subfolder_path):
        ds_store_path = os.path.join(subfolder_path, '.DS_Store')
        if os.path.exists(ds_store_path):
            os.remove(ds_store_path)
        for filename in sorted(os.listdir(subfolder_path), key=numerical_sort):
            # print(subfolder, filename)
            src_image_path = os.path.join(subfolder_path, filename)
            # print(src_image_path)

            image = Image.open(src_image_path).convert('RGB')
            image = resizePadding(image, opt.imgW, opt.imgH)

            if torch.cuda.is_available():
                image = image.cuda()

            image = image.view(1, *image.size())
            image = Variable(image)

            start_time = time.time()
            preds = model(image)

            values, prob = softmax(preds, dim=-1).max(2)
            preds_idx = (prob > 0).nonzero()
            sent_prob = values[preds_idx[:, 0], preds_idx[:, 1]].mean().item()

            _, preds = preds.max(2)
            preds = preds.transpose(1, 0).contiguous().view(-1)
            preds_size = Variable(torch.IntTensor([preds.size(0)]))
            raw_pred = converter.decode(preds.data, preds_size.data, raw=True)
            sim_pred = converter.decode(preds.data, preds_size.data, raw=False)

            df = pd.DataFrame(
                [[f'{subfolder}/{filename}', sim_pred]], columns=['id', 'answer'])
            data = pd.concat([data, df])

            # print('%-20s => %-20s : prob: %s time: %s' %
            #       (raw_pred, sim_pred, sent_prob, time.time() - start_time))
    print(f'\u2705 {subfolder}')
data.to_csv('submit.csv', index=False)
