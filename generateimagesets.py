import os
import random
import glob

trainval_percent = 1
train_percent = 0.8

xmlfilepath = 'datasets/dhaka-ai/voc/Annotations'
txtsavepath = 'datasets/dhaka-ai/voc/ImageSets/Main/'
total_xml = glob.glob(os.path.join(xmlfilepath, '*.xml'))

num = len(total_xml)
list = range(num)
tv = int(num*trainval_percent)
tr = int(tv*train_percent)
trainval = random.sample(list, tv)
train = random.sample(trainval, tr)
ftrainval = open('datasets/dhaka-ai/voc/ImageSets/Main/trainval.txt', 'w')
ftest = open('datasets/dhaka-ai/voc/ImageSets/Main/test.txt', 'w')
ftrain = open('datasets/dhaka-ai/voc/ImageSets/Main/train.txt', 'w')
fval = open('datasets/dhaka-ai/voc/ImageSets/Main/val.txt', 'w')

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
