#-*-coding:utf-8 -*-
import cv2
import numpy as np
from PIL import Image
import imutils
from imutils import contours
from imutils.perspective import four_point_transform
import os
import json
from PIL import Image,ImageDraw,ImageOps
from rec_ans_area_base import RecAnsAreaBase
from detect_image_std import ImageDetect

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

class RecAnsArea:
    """识别出试卷上的类答题卡区域
    """

    def __init__(self,image_path="test.jpg",paperno=None):
        self.paperno = paperno
        self.image_path = image_path
        self.image = cv2.imread(self.image_path)
        #膨胀操作后的图片
        self.dilate_image_path = None
        #腐蚀操作后的图片
        self.erode_image_path = None
        #反色后的图片
        self.inverse_image_path = None

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

    
    def get_std_point(self):
        """
        """
        img_dect = ImageDetect(self.image_path)
        #img_dect._init_dilate()
        img_dect.closeopration()
        img_dect.close_thresh()
        std_point = img_dect.get_std_point()
        return std_point
    


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
        self.dilate_image_path = close_path
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


    def closeopration(self):  
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10))  

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
        self._erode_dilate(w,h)

    def _close(self,w=5,h=10):
        """
        """
        #自定义闭操作
        self._dilate_erode(w,h)


    def cut_ans_area(self,cut_point=None):
        '''
        从试卷裁剪出答题卡区域图片
        '''
        img = Image.open(self.image_path)
        region = img.crop(cut_point)

        spt_lst = os.path.splitext(self.image_path)
        cut_path = spt_lst[0] + '_cut_'+str(cut_point)+ spt_lst[1]

        region.save(cut_path)

        return cut_path

    def _dilate_for_inverse(self,ans_area_path=None,w=3,h=3):
        '''
        图像膨胀操作
        '''
        inverse_image = cv2.imread(ans_area_path)
        inverse_gray = cv2.cvtColor(inverse_image, cv2.COLOR_BGR2GRAY)
        #对灰度图进行二值化处理
        ret,inverse_thresh = cv2.threshold(inverse_gray,0,250,cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

        dilsize = cv2.getStructuringElement(cv2.MORPH_RECT,(w,h))
        inverse_dilate_image = cv2.dilate(inverse_thresh,dilsize)
        #输出膨胀操作后的图像
        spt_lst = os.path.splitext(self.image_path)
        inverse_dilate_image_path = spt_lst[0] + '_dilate' + spt_lst[1]
        cv2.imwrite(inverse_dilate_image_path,inverse_dilate_image)
        return inverse_dilate_image_path 




    def get_inverse_image(self,org_ans_path):
        """获取反色处理后的图片
        """
        print 11111111111111
        #输出膨胀操作后的图像
        spt_lst = os.path.splitext(org_ans_path)
        inverse_path = spt_lst[0] + '_inverse' + spt_lst[1]
        inverse_image = self._dilate_for_inverse(org_ans_path)
        im02 = Image.open(inverse_image)
        im = ImageOps.invert(im02)
        im.save(inverse_path)
        self.inverse_image_path = inverse_path
        return self.inverse_image_path


    def get_ans_path(self,std_point=None):
        """识别出答案轮廓
        """
        dilate_image = self._dilate(5,5)
        cnts = cv2.findContours(dilate_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cv2.findContours(dilate_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        cnts = cv2.findContours(dilate_image, cv2.RETR_EXTERNAL, cv2.RETR_CCOMP)
        all_attr = cnts[1]
        cnts = cnts[0] if imutils.is_cv2() else cnts[1]

        ans_area_path = None
        if len(cnts) > 0:
            cnts = sorted(cnts, key=cv2.contourArea, reverse=True)

            all_c = []
            all_cnts_dct = {}
            for c in cnts:
                peri = 0.01*cv2.arcLength(c, True)

                approx = cv2.approxPolyDP(c, peri, True)
                (x, y, w, h) = cv2.boundingRect(c)

                if w > self.width/5 and x>100 and y>100 and h>100:

                    if x>self.width/2 and w<self.width/2:
                        continue

                    #标识出识别出的机型轮廓并输出
                    cv2.drawContours(self.image,c,-1,(255,0,255),3)
                    spt_lst = os.path.splitext(self.image_path)
                    draw_path = spt_lst[0] + '_draw' + spt_lst[1]
                    cv2.imwrite(draw_path,self.image)

                    all_c.append({'x':x,'y':y,'w':w,'h':h})

                    #设置切割点
                    cut_point = (x,y,x+w,y+h)
                    #左、右上、右下
                    org_cut_point = ((x,y),(x+w,y),(x+w,y+h))
                    #这里会导致XZB2018RJHXB02004答题卡轮廓识别错误
                    #cut_point = (x+5,y+5,x+w-10,y+h-10)
                    if self.get_std_point():
                        print self.get_std_point(),9999999999999999999999999
                        print 11111111111111111111
                        add_w = self.get_std_point()["w"]*1.5
                        add_h = self.get_std_point()["h"]
                        #cut_point = (x-add_w,y-add_h,x+w+add_w,y+h+add_h)
                        cut_point = (x,y,x+w+add_w,y+h+add_h)
                    else:
                        cut_point = (x-60,y-50,x+w+60,y+h+50)

                    cut_path = self.cut_ans_area(cut_point)
                    reaab = RecAnsAreaBase(cut_path)
                    base_cnts_count = reaab.rec_cnts()

                    all_cnts_dct[base_cnts_count] = cut_path,org_cut_point

            #从统计中找出轮廓数最多的作为答题卡
            #mylog(base_cnts_count=base_cnts_count)
            keys = all_cnts_dct.keys()
            keys.reverse()
            try:
                max_key = keys[0]
                ans_area_path = all_cnts_dct.get(max_key)[0]
                org_cut_point = all_cnts_dct.get(max_key)[1]
            except:
                ans_area_path = None
                org_cut_point = None

        #获取答案膨胀反色图片
        print ans_area_path,6666666666666666666666666666666666666666666
        self.inverse_image_path = self.get_inverse_image(ans_area_path) 
        print self.inverse_image_path,77777777777777777777
        return ans_area_path,org_cut_point,self.inverse_image_path,self.get_std_point()



def main():
    img_dect = RecAnsArea("test.jpg")
    ans_area_path,org_cut_point,inverse_image_path,std_point = img_dect.get_ans_path()


if __name__ == "__main__":
    main()
