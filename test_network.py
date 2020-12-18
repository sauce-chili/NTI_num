import numpy as np
from Network import *
import cv2

img = cv2.imread('pictures_numeric_val/20200413001836_5072.jpg')
net = Net('numModel_V4.h5')

images = net.imageProcessing(img)

im_buff = []
vec = []
for im in images:
    result = net.predict(im)
    acc = net.getAccuriry()
    if result is not 0:
        if result not in vec:
            im_buff.append(im)
            vec.append(result)

print(vec)

while True:
    cv2.imshow('image', img)
    #cv2.imshow('1', im_buff[0])
    #cv2.imshow('2', im_buff[1])
    # cv2.imshow('3',im_buff[2])
    if cv2.waitKey(1) & 0xFF == ord('f'):
        break
