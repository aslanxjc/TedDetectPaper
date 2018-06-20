#!/usr/bin/env python
# -*- coding:utf-8 -*-
# libs
import cv2
import imutils
import numpy as np
from imutils import contours
from imutils.perspective import four_point_transform
from PIL import Image
import os

ANSWER_KEY = {0: 1, 1: 4, 2: 0, 3: 3, 4: 1}


def get_std_point(img_path=None):
    '''
    识别出试卷上的两个定位点
    '''
    tmp = []
    print img_path

    image = cv2.imread(img_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blurred, 75, 200)
    #edged = cv2.Canny(gray, 75, 200)
    cv2.imwrite('get_point.png',edged)
    #cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #cnts = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    #cnts = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cv2.findContours(edged.copy(), cv2.RETR_CCOMP , cv2.CHAIN_APPROX_SIMPLE)
    all_attr = cnts[1]
    cnts = cnts[0] if imutils.is_cv2() else cnts[1]

    docCnt = 0

    if len(cnts) > 0:
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)

        for i,c in enumerate(cnts):
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)
            #if len(approx) == 4:
            (x, y, w, h) = cv2.boundingRect(c)
            if w>30 and w<40 and h>40 and h<50:
                print x,y,w,h
                cv2.drawContours(image,c,-1,(255,0,255),3)
                cv2.imwrite('mark_std_point.png',image)
                tmp.append((x,y))

        new_tmp = list(set(tmp))
        new_tmp.sort()
        #print new_tmp
        return new_tmp

    return []

    return 0,0,0,0,0,0


def cut_img(img_path='mark.jpg',point=(),newname='real_new.jpg'):
    '''
    '''
    from PIL import Image,ImageDraw
    img=Image.open(img_path)  #打开图像
    box = point 
    region = img.crop(box)
    region.save(newname)

    return newname


def CalculateLineAngle(x,y,xx,yy):
    '''
    '''
    import math
    xdis = xx-x
    ydis = yy-y
    angle = math.atan2(xdis, ydis)
    angle = angle / math.pi *180 

    return angle

def rotate(imgname="qingxie.jpg",angle=0):
    '''
    '''
    new = "newqingxie.jpg"
    img = Image.open(imgname)
    ang = int(angle)
    img.rotate(ang).save(imgname)
    return imgname
    return new


def rec_rotate(path=None):
    '''
    '''
    from detect_image_std import get_std_point
    plist = get_std_point(path,True)

    new_tmp = []
    for _p in plist:
        x=_p.get("x")
        y=_p.get("y")
        new_tmp.append((x,y))


    new_tmp = list(set(new_tmp))
    new_tmp.sort()
    plist = new_tmp

    print plist,444444444444444444444
    angle = CalculateLineAngle(plist[1][0],plist[1][1],plist[0][0],plist[0][1])
    print angle,88888888888888888888888888
    angle = -(90+angle)
    #angle = 90+angle
    new_path = rotate(path,angle)
    print new_path,55555555555555
    nplist = get_std_point(new_path)
    print nplist,66666666666666666666

    #cut_name = os.path.splitext(path)[0]+'_cut' + os.path.splitext(path)[1]

    #print cut_name,8888888888888888888

    #cut_img(new_path,(nplist[0][0],nplist[0][1],nplist[3][0]+20,nplist[3][1]+40),cut_name)

    #return cut_name


if __name__ == "__main__":
    #找到矩形答题卡轮廓
    rec_rotate('qingxie.jpg')
    #get_std_point('qingxie.jpg')
    
