#!/usr/bin/env python

from glob import glob
import os                                                       
import cv2 

dir = "datasets/dhaka-ai/voc/JPEGImages/"

pngs = glob(os.path.join(dir, "*.png"))

for j in pngs:
    img = cv2.imread(j)
    cv2.imwrite(j[:-3] + 'jpg', img)
    os.remove(j)



pngs = glob(os.path.join(dir, "*.PNG"))

for j in pngs:
    img = cv2.imread(j)
    cv2.imwrite(j[:-3] + 'jpg', img)
    os.remove(j)

