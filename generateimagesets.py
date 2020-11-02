#!/usr/bin/env python

import os
import random
import glob

trainval_percent = 0.95
train_percent = 0.90


xmlfilepath = 'datasets/dhaka-ai/voc/Annotations'
txtsavepath = 'datasets/dhaka-ai/voc/ImageSets/Main/'

total_xml = glob.glob(os.path.join(xmlfilepath, '*.xml'))

num = len(total_xml)
list = range(num)
tv = int(num*trainval_percent)
tr = int(tv*train_percent)
trainval = random.sample(list, tv)
train = random.sample(trainval, tr)
ftrainval = open(os.path.join(txtsavepath, 'trainval.txt'), 'w')
ftest = open(os.path.join(txtsavepath, 'test.txt'), 'w')
ftrain = open(os.path.join(txtsavepath, 'train.txt'), 'w')
fval = open(os.path.join(txtsavepath, 'val.txt'), 'w')

for i in list:
    name = os.path.basename(total_xml[i])[:-4]+'\n'
    if i in trainval:
        ftrainval.write(name)
        if i in train:
            ftrain.write(name)
        else:
            fval.write(name)
    else:
        ftest.write(name)

ftrainval.close()
ftrain.close()
fval.close()
ftest.close()
