#-*-coding:utf-8 -*-
import cv2
import numpy as np
from PIL import Image
import imutils
from imutils import contours
from imutils.perspective import four_point_transform
import os
import json
from collections  import OrderedDict
from PIL import Image,ImageDraw

from detect_image_std import get_std_point

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

class CloseImage:

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

    def _dilate(self,w=5,h=5):
        '''
        图像膨胀操作
        '''
        dilsize = cv2.getStructuringElement(cv2.MORPH_RECT,(w,h))
        self.image = cv2.dilate(self.image,dilsize)
        #输出膨胀操作后的图像
        spt_lst = os.path.splitext(self.image_path)
        close_path = spt_lst[0] + 'close_dilate' + spt_lst[1]
        cv2.imwrite(close_path,self.image)
        self.dilate_image_path = close_path
        return self.image

    def rec_kh_std_line(self,path=None):
        '''
        直线检测
        '''
        lines_data = []
        minLineLength = self.height/3*2
        maxLineGap = 100
        
        lines = cv2.HoughLinesP(self.thresh,1,np.pi,118,minLineLength,maxLineGap)
        for line in lines[:,:,:]:
            _tmp_dct = {}
            print line[0,:]
            x2,y2,x1,y1 = line[0,:]
            #print x1,y1,x2,y2
            _tmp_dct["x1"] = x1
            _tmp_dct["y1"] = y1
            _tmp_dct["x2"] = x2
            _tmp_dct["y2"] = y2
            _tmp_dct["h"] = abs(y2-y1)
            
            lines_data.append(_tmp_dct)

            cv2.line(self.image,(x1,y1),(x2,y2),(0,255,0),1)
            #输出答题卡区域图像
            ans_lst = os.path.splitext(self.image_path)
            ans_path = ans_lst[0] + '_line' + ans_lst[1]
            cv2.imwrite(ans_path,self.image)


        #考号标准线识别
        lines_data = sorted(lines_data,key=lambda x:x["x1"])

        return lines_data

        tmp_dct = OrderedDict()
        for _line in lines_data:
            x1 = _line.get("x1")
            y1 = _line.get("y1")
            x2 = _line.get("x2")
            y2 = _line.get("y2")

            if not tmp_dct:
                tmp_dct[x1] = [_line]
            else:
                if (x1-tmp_dct.keys()[-1]) < 10:
                    tmp_dct[tmp_dct.keys()[-1]].append(_line)
                else:
                    tmp_dct[x1] = [_line]

        #mylog(tmp_dct=dict(tmp_dct))
        for i,tpl in enumerate(tmp_dct.items()):
            lines_list = tpl[1]
            print lines_list,555555555555555555
            lines_list = sorted(lines_list,key=lambda x:x["h"],reverse=True)
            #return lines_list
            if i == 4:
                return lines_list[0]

        return {}

        return lines_data


    def closeopration(self,w=18,h=15):  
        #self._dilate()
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (w, h))  
        self.iClose = cv2.morphologyEx(self.image, cv2.MORPH_CLOSE, kernel)  
        #输出闭操作后的图像
        spt_lst = os.path.splitext(self.image_path)
        close_path = spt_lst[0] + '_close' + spt_lst[1]
        cv2.imwrite(close_path,self.iClose)
        return close_path  




class CloseCntsDetect:

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



    def get_cnts(self,std_point=None):
        """识别出答案轮廓
        """

        #cnts = cv2.findContours(self.dilate_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cv2.findContours(self.thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        all_attr = cnts[1]
        cnts = cnts[0] if imutils.is_cv2() else cnts[1]

        all_c = []
        if len(cnts) > 0:
            cnts = sorted(cnts, key=cv2.contourArea, reverse=True)

            for c in cnts:
                peri = 0.01*cv2.arcLength(c, True)

                approx = cv2.approxPolyDP(c, peri, True)
                (x, y, w, h) = cv2.boundingRect(c)

                all_c.append({'x':x,'y':y,'w':w,'h':h})
            #mylog(all_c=all_c)
            all_c = sorted(all_c,key=lambda x:x["w"],reverse=True)
        return all_c


def get_ans_area_cnts(inverse_image_path="test_cut_(543, 2219, 1860, 2603)_inverse.jpg",
        std_point={"w":37,"h":47},std_quenos=[]):
    """从答题卡中识别出图像识别技术清洗后的轮廓包含考号和答案
    """
    w =std_point.get("w")
    h =std_point.get("h")
    w = 37
    h = 48
    #闭操作清除杂质
    image = CloseImage(inverse_image_path)
    kh_std_line = image.rec_kh_std_line()
    if len(std_quenos) > 20:
        cimage_path = image.closeopration(int(w/2.5),int(h/2.5))
    else:
        cimage_path = image.closeopration(int(w/2.5),int(h/2.5))
    #从闭操作后的图像提取轮廓
    ccimage = CloseCntsDetect(cimage_path)
    cnts = ccimage.get_cnts()
    #mylog(cnts=cnts)

    return cnts,kh_std_line


def main():
    get_ans_area_cnts()
    pass



if __name__ == "__main__":
    main()
