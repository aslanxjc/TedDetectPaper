#-*-coding:utf-8 -*-
import os
import sys
import cv2
import collections
from new_rec_ans_area import RecAnsArea
from rec_fill_cnts import get_ans_area_cnts
from new_rec_cnts_filter import CntsFilter
import numpy as np


def mylog(**kargs):
    '''
    格式化打印数据
    '''
    k = kargs.keys()[0]
    from pprint import pprint
    print '\n\n-----------------------beigin----------------'
    #pprint '{}s:{}'.format((k,kargs.get(k)))
    pprint('{}:'.format(k))
    pprint(kargs.get(k))
    print '----------------------end--------------------\n\n'


class DetectPaper:

    def __init__(self,path="test.jpg"):
        """
        """
        self.paper_path = path
        self.ans_area_path = None
        self.org_cut_point = None
        self.inverse_image_path = None

    def get_ans_area_path(self):
        """获取答案区域
        """
        raaobj = RecAnsArea(self.paper_path)
        self.ans_area_path,\
            self.org_cut_point,\
                self.inverse_image_path = raaobj.get_ans_path()

        #反色处理好的答题卡区域图片
        self.image = cv2.imread(self.inverse_image_path)
        #图片转成灰度图
        self.gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        #对灰度图进行二值化处理
        ret,thresh = cv2.threshold(self.gray,0,250,cv2.THRESH_BINARY_INV \
                        | cv2.THRESH_OTSU)
        self.thresh  = thresh

    def rec_all_fill_cnts(self):
        """
        """
        self.get_ans_area_path()
        all_cnts = get_ans_area_cnts(self.inverse_image_path)
        mylog(all_cnts=all_cnts)
        return all_cnts

    def get_roi(self,rec=(1192,347,1237,369)):
        """指定矩形角点提取感兴趣矩形区域
        """
        mask = np.zeros(self.thresh.shape[0:2],dtype=np.uint8) 
        cv2.rectangle(sss,(rec[0],rec[1]),(rec[2],rec[3]),255,-1)

        mask_roi=cv2.bitwise_or(self.thresh, self.thresh,mask=mask)
        #输出二值化操作后的图像
        spt_lst = os.path.splitext(self.inverse_image_path)
        mask_path = spt_lst[0] + '_mask_roi' + spt_lst[1]
        cv2.imwrite(mask_path,sss)
        return mask_roi

    def get_roi_points(self,std_x_points,std_y_points):
        """通过stdxy计算目标roi区域坐标
        """
        mask = np.zeros(self.thresh.shape[0:2],dtype=np.uint8) 

        std_y_points = sorted(std_y_points,key=lambda x:x["y"],reverse=True)
        for i,_xp in enumerate(std_x_points):
            _xx = _xp.get("x")
            _xy = _xp.get("y")
            _xw = _xp.get("w")
            _xh = _xp.get("h")
            for j,_yp in enumerate(std_y_points):
                _yx = _yp.get("x")
                _yy = _yp.get("y")
                _yw = _yp.get("w")
                _yh = _yp.get("h")
                #
                #_roi_point = (_xx,_yy,_xx+_xw,_yy+_yh)
                rec = (_xx,_yy,_xx+_xw,_yy+_yh)

                cv2.rectangle(mask,(rec[0],rec[1]),(rec[2],rec[3]),255,-1)

        mask_roi=cv2.bitwise_or(self.thresh, self.thresh,mask=mask)
        #输出二值化操作后的图像
        spt_lst = os.path.splitext(self.inverse_image_path)
        mask_path = spt_lst[0] + '_mask_roi' + spt_lst[1]
        cv2.imwrite(mask_path,mask_roi)

    def rec_paper(self):
        """
        """
        all_cnts = self.rec_all_fill_cnts()
        cntsfilter = CntsFilter(all_cnts,self.org_cut_point)
        self.std_x_points,self.std_y_points,self.kh_points = cntsfilter.sep_std_cnts()
        mylog(std_x_points=self.std_x_points)
        mylog(std_y_points=self.std_y_points)
        mylog(std_y_points=self.kh_points)
        self.get_roi_points(self.std_x_points,self.std_y_points)


class TestDetectPaper:

    def __init__(self,image_path="test_cut_(544, 630, 1861, 1091)_inverse.jpg"):
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

    def get_roi(self,rec=(1192,347,1237,369)):
        """指定矩形角点提取感兴趣矩形区域
        """
        print self.thresh.shape[0:2]
        print np.shape(self.image)

        #sss=np.zeros(self.thresh.shape[0:2],dtype=np.uint8) 
        sss=np.zeros(self.thresh.shape[0:2],dtype=np.uint8) 
        #sss=np.zeros([1320,460],dtype=np.uint8) 
        #sss[rec[0]:rec[1],rec[2]:rec[3]]=255
        cv2.rectangle(sss,(rec[0],rec[1]),(rec[2],rec[3]),255,-1)

        #输出二值化操作后的图像
        spt_lst = os.path.splitext(self.image_path)
        mask_path = spt_lst[0] + '_mask' + spt_lst[1]
        cv2.imwrite(mask_path,sss)

        mask_image=cv2.bitwise_or(self.thresh, self.thresh,mask=sss)
        total = cv2.countNonZero(mask_image)
        print total,222222222222222

        #输出二值化操作后的图像
        spt_lst = os.path.splitext(self.image_path)
        mask_path = spt_lst[0] + '_mask_area_and' + spt_lst[1]
        cv2.imwrite(mask_path,mask_image)



if __name__ == "__main__":
    dpobj = DetectPaper()
    dpobj.rec_all_fill_cnts()
    dpobj.rec_paper()
    #tdpobj = TestDetectPaper()
    #tdpobj.get_roi()

