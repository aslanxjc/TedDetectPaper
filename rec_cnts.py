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

class RecCnts:
    """识别出试卷上的类答题卡区域
    """

    def __init__(self,image_path="test.jpg"):
        self.image_path = image_path
        self.image = cv2.imread(self.image_path)

        #图片的宽高度
        self.width = self.image.shape[1]
        self.height = self.image.shape[0]

        #图片转成灰度图
        self.gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)

        #边缘检测后的图
        self.edged = cv2.Canny(self.gray, 75, 200)

        #对灰度图进行二值化处理
        ret,thresh = cv2.threshold(self.gray,0,250,cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
        self.thresh  = thresh
        #输出二值化操作后的图像
        spt_lst = os.path.splitext(self.image_path)
        close_path = spt_lst[0] + '_thresh' + spt_lst[1]
        cv2.imwrite(close_path,self.thresh)

    def _erode(self,w=5,h=10):
        '''
        图像腐蚀操作
        '''
        #腐蚀核大小
        edsize = cv2.getStructuringElement(cv2.MORPH_RECT,(w,h))
        self.erode_image = cv2.erode(self.thresh,edsize)
        #输出腐蚀操作后的图像
        spt_lst = os.path.splitext(self.image_path)
        close_path = spt_lst[0] + '_erode' + spt_lst[1]
        cv2.imwrite(close_path,self.erode_image)
        return self.erode_image

    def _erode_dilate(self,w=5,h=10):
        '''
        图像腐蚀操作
        '''
        #腐蚀核大小
        edsize = cv2.getStructuringElement(cv2.MORPH_RECT,(w,h))
        self.erode_dilate_image = cv2.erode(self.dilate_image,edsize)
        #输出腐蚀操作后的图像
        spt_lst = os.path.splitext(self.image_path)
        close_path = spt_lst[0] + '_erode_dilate' + spt_lst[1]
        cv2.imwrite(close_path,self.erode_dilate_image)
        return self.erode_dilate_image


    def _dilate(self,w=5,h=10):
        '''
        图像膨胀操作
        '''
        dilsize = cv2.getStructuringElement(cv2.MORPH_RECT,(w,h))
        self.dilate_image = cv2.dilate(self.thresh,dilsize)
        #输出膨胀操作后的图像
        spt_lst = os.path.splitext(self.image_path)
        close_path = spt_lst[0] + '_dilate' + spt_lst[1]
        cv2.imwrite(close_path,self.dilate_image)
        return self.dilate_image

    def _dilate_erode(self,w=5,h=10):
        '''
        图像膨胀操作
        '''
        dilsize = cv2.getStructuringElement(cv2.MORPH_RECT,(w,h))
        self.dilate_erode_image = cv2.dilate(self.erode_dilate_image,dilsize)
        #输出膨胀操作后的图像
        spt_lst = os.path.splitext(self.image_path)
        close_path = spt_lst[0] + '_dilate_erode' + spt_lst[1]
        cv2.imwrite(close_path,self.dilate_erode_image)
        return self.dilate_erode_image


    def closeopration(self):  
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10))  
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (self.width/2, 1))  

        self.iClose = cv2.morphologyEx(self.image, cv2.MORPH_CLOSE, kernel)  
        #输出闭操作后的图像
        spt_lst = os.path.splitext(self.image_path)
        close_path = spt_lst[0] + '_close' + spt_lst[1]
        cv2.imwrite(close_path,self.iClose)
        return self.iClose  


    def close_thresh(self):
        self.close_gray = cv2.cvtColor(self.iClose, cv2.COLOR_BGR2GRAY)
        self.close_thresh = cv2.threshold(self.close_gray.copy(),200,255,cv2.THRESH_BINARY_INV)[1]
        #输出闭操作后的图像
        spt_lst = os.path.splitext(self.image_path)
        close_path = spt_lst[0] + '_close_thresh' + spt_lst[1]
        cv2.imwrite(close_path,self.close_thresh)


    def _open(self,w=15,h=10):
        """
        """
        #先膨胀
        self._dilate(3,2)
        #自定义开操作
        self.open_image = self._erode_dilate(w,h)
        #输出闭操作后的图像
        spt_lst = os.path.splitext(self.image_path)
        close_path = spt_lst[0] + '_open' + spt_lst[1]
        cv2.imwrite(close_path,self.open_image)
        return self.open_image 

    def _close(self,w=5,h=10):
        """
        """
        #自定义闭操作
        self.close_image = self._dilate_erode(w,h)
        #输出闭操作后的图像
        spt_lst = os.path.splitext(self.image_path)
        close_path = spt_lst[0] + '_close' + spt_lst[1]
        cv2.imwrite(close_path,self.close_image)
        return self.close_image 


    def get_cnts(self,std_point=None):
        """识别出答案轮廓
        """
        self._open()
        close = self._close()

        cnts = cv2.findContours(self.close_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        all_attr = cnts[1]
        cnts = cnts[0] if imutils.is_cv2() else cnts[1]

        all_c = []
        if len(cnts) > 0:
            cnts = sorted(cnts, key=cv2.contourArea, reverse=True)

            all_c = []
            for c in cnts:
                #print c,111111111111111
                peri = 0.01*cv2.arcLength(c, True)

                approx = cv2.approxPolyDP(c, peri, True)
                (x, y, w, h) = cv2.boundingRect(c)
                if len(approx)==4:
                    (x, y, w, h) = cv2.boundingRect(c)
                    if w > self.width/5:
                        ##标识出识别出的机型轮廓并输出
                        #cv2.drawContours(self.image,c,-1,(255,0,255),3)
                        #spt_lst = os.path.splitext(self.image_path)
                        #draw_path = spt_lst[0] + '_draw' + spt_lst[1]
                        #cv2.imwrite(draw_path,self.image)
                        pass
                    #all_c.append({'x':x,'y':y,'w':w,'h':h})
                all_c.append({'x':x,'y':y,'w':w,'h':h})
            mylog(all_c=all_c)
        return all_c

def main():
    img_dect = RecCnts("test.png")
    #先膨胀
    img_dect.get_cnts()


if __name__ == "__main__":
    main()
