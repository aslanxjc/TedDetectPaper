#-*-coding:utf-8 -*-
import cv2
import numpy as np
from PIL import Image
import imutils
from imutils import contours
from imutils.perspective import four_point_transform
import os
import json
from PIL import Image,ImageDraw

def mylog(**kargs):
    '''
    '''
    k = kargs.keys()[0]
    from pprint import pprint
    print '\n\n\n\n-----------------------beigin----------------'
    #pprint '{}s:{}'.format((k,kargs.get(k)))
    pprint('{}:'.format(k))
    pprint(kargs.get(k))
    print '----------------------end--------------------\n\n\n\n'


class ImageDetect:

    def __init__(self,image_path="qingxie.jpg",quenos=None,std_ans_y=None):
        '''
        '''

        self.quenos = quenos
        self.std_ans_y = std_ans_y

        self.image_path = image_path
        self.erode_image = None
        self.dilate_image = None
        self.kh_std_x = None
        self.kh_std_y = None
        self.kh_std_w = None
        self.kh_std_h = None
        self.edsize = cv2.getStructuringElement(cv2.MORPH_RECT,(15,20))
        self.edsize1 = cv2.getStructuringElement(cv2.MORPH_RECT,(15,20))

        self.init_dilsize = cv2.getStructuringElement(cv2.MORPH_RECT,(2,2))

        self.image = cv2.imread(self.image_path)
        self.image_obj = Image.open(self.image_path)

        #图片的宽度
        self.width = self.image.shape[1]
        self.height = self.image.shape[0]

        #设置切割点
        cut_box = (0,0,self.width*0.875,self.height*0.125)
        cut_box = (self.width*0.875,0,self.width,self.height*0.125)
        cut_box = (0,0,self.width,self.height*0.125)
        region = self.image_obj.crop(cut_box)
        image_path_split = os.path.splitext(os.path.abspath(self.image_path))
        cut_image_path = image_path_split[0]+'_cut_std_point'+image_path_split[1]
        region.save(cut_image_path)
        self.image = cv2.imread(cut_image_path)

        self.gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(self.gray, (5, 5), 0)
        self.edged = cv2.Canny(self.gray, 75, 200)
        self.thresh = cv2.threshold(self.gray.copy(),200,255,cv2.THRESH_BINARY_INV)[1]
        spt_lst = os.path.splitext(image_path)
        edged_path = spt_lst[0] + '_edged' + spt_lst[1]
        cv2.imwrite(edged_path,self.edged)

    def _erode(self,img=None):
        '''
        图像腐蚀操作
        '''
        self.erode_image = cv2.erode(self.thresh,self.edsize)
        spt_lst = os.path.splitext(self.image_path)
        erode_path = spt_lst[0] + '_erode' + spt_lst[1]
        cv2.imwrite(erode_path,self.erode_image)
        pass

    def closeopration(self):  
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 30))  
        self.iClose = cv2.morphologyEx(self.image, cv2.MORPH_CLOSE, kernel)  
        spt_lst = os.path.splitext(self.image_path)
        close_path = spt_lst[0] + '_close' + spt_lst[1]
        cv2.imwrite(close_path,self.iClose)
        return self.iClose  

    def close_thresh(self):
        self.close_gray = cv2.cvtColor(self.iClose, cv2.COLOR_BGR2GRAY)
        self.close_thresh = cv2.threshold(self.close_gray.copy(),200,255,cv2.THRESH_BINARY_INV)[1]
        spt_lst = os.path.splitext(self.image_path)
        close_path = spt_lst[0] + 'cut_std_point_close_thresh' + spt_lst[1]
        cv2.imwrite(close_path,self.close_thresh)


    def _dilate(self):
        '''
        图像膨胀操作
        '''
        self.dilate_image = cv2.erode(self.thresh,self.edsize1)
        spt_lst = os.path.splitext(self.image_path)
        dilate_path = spt_lst[0] + '_dilate' + spt_lst[1]
        cv2.imwrite(dilate_path,self.dilate_image)

    def _init_dilate(self):
        '''
        图像膨胀操作
        '''
        self.init_dilate_image = cv2.dilate(self.thresh,self.init_dilsize)
        spt_lst = os.path.splitext(self.image_path)
        dilate_path = spt_lst[0] + '_dilate' + spt_lst[1]
        cv2.imwrite(dilate_path,self.init_dilate_image)


    def get_std_point(self):
        '''
        提取已填涂轮廓
        '''
        img,cnts,_ = cv2.findContours(self.close_thresh, cv2.RETR_EXTERNAL , cv2.CHAIN_APPROX_SIMPLE)

        kh_lst = []
        dan_lst = []
        all_lst = []
        for c in cnts:
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.1* peri, True)
            approx_length = cv2.arcLength(approx,True)

            (x, y, w, h) = cv2.boundingRect(c)
            all_lst.append({'x':x,'y':y,'w':w,'h':h})
        all_lst = sorted(all_lst,key=lambda x:x['x'])
        if all_lst:
            tmp = []
            mylog(all_lst=all_lst)
            for c in all_lst:
                h = c.get('h')
                w = c.get('w')
                x = c.get('x')
                y = c.get('y')
                if float(h)/w < 0.5 or float(h)/w > 1.95 or x<10 or y<10:
                    continue
                else:
                    mylog(c=c)
                    tmp.append(c)
            std_point = tmp[0] if tmp else None
            return std_point
            return tmp[0]
        else:
            return None


def get_std_point(img_path='test.jpg',):
    '''
    '''
    img_dect = ImageDetect(img_path)
    img_dect.closeopration()
    img_dect.close_thresh()
    std_point = img_dect.get_std_point()
    return std_point

def get_std_point_1(img_path='test.jpg',):
    '''
    '''
    img_dect = ImageDetect(img_path)
    img_dect.closeopration()
    img_dect.close_thresh()
    img_dect._erode()
    img_dect._dilate()
    std_point = img_dect.get_std_point()
    return std_point


if __name__ == "__main__":
    print get_std_point()

