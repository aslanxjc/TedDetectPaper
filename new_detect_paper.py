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

    def __init__(self,path="test.jpg",quenos=[]):
        """
        """
        self.paper_path = path
        self.quenos = quenos
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
        #mylog(all_cnts=all_cnts)
        return all_cnts

    def get_roi(self,rec=(1192,347,1237,369)):
        """指定矩形角点提取感兴趣矩形区域
        """
        mask = np.zeros(self.thresh.shape[0:2],dtype=np.uint8) 
        cv2.rectangle(mask,(rec[0],rec[1]),(rec[2],rec[3]),255,-1)

        mask_roi=cv2.bitwise_or(self.thresh, self.thresh,mask=mask)
        #输出二值化操作后的图像
        spt_lst = os.path.splitext(self.inverse_image_path)
        mask_path = spt_lst[0] + '_mask_roi' + spt_lst[1]
        cv2.imwrite(mask_path,mask_roi)
        return mask_roi

    def get_std_pixels(self,std_x_points=[]):
        """获取标准点总像素点
        """
        std_cnts = std_x_points[0]
        x = std_cnts.get("x")
        y = std_cnts.get("y")
        w = std_cnts.get("w")
        h = std_cnts.get("h")
        std_point = (x,y,x+w,y+h)
        mask_std_x = self.get_roi(std_point)
        std_pixels = cv2.countNonZero(mask_std_x)
        return std_pixels

    def get_roi_points(self,std_x_points,std_y_points):
        """通过stdxy计算目标roi区域坐标
        """
        roi_points_dct = collections.OrderedDict()
        mask = np.zeros(self.thresh.shape[0:2],dtype=np.uint8) 
        #std_y_points = sorted(std_y_points,key=lambda x:x["y"],reverse=True)
        for i,_xp in enumerate(std_x_points):
            _xx = _xp.get("x")
            _xy = _xp.get("y")
            _xw = _xp.get("w")
            _xh = _xp.get("h")
            ans_chos = []
            for j,_yp in enumerate(std_y_points):
                _yx = _yp.get("x")
                _yy = _yp.get("y")
                _yw = _yp.get("w")
                _yh = _yp.get("h")
                #roi坐标点
                _roi_point = (_xx,_yy,_xx+_xw,_yy+_yh)
                rec = (_xx,_yy,_xx+_xw,_yy+_yh)
                ans_chos.append(_roi_point)

            roi_points_dct[i] = ans_chos

        return roi_points_dct

    def rec_paper(self):
        """
        """
        all_cnts = self.rec_all_fill_cnts()
        cntsfilter = CntsFilter(all_cnts,self.quenos,self.org_cut_point)
        self.std_x_points,self.std_y_points,self.kh_points = cntsfilter.sep_std_cnts()
        mylog(std_x_points=self.std_x_points)
        mylog(std_y_points=self.std_y_points)
        mylog(kh_points=self.kh_points)
        roi_points_dct = self.get_roi_points(self.std_x_points,self.std_y_points)
        ###########################
        #答案识别
        #计算roi非0像素点
        std_pixels = self.get_std_pixels(self.std_x_points)
        ans_options = {0:"A",1:"B",2:"C",3:"D",4:"E",5:"F",6:"G"}
        ans_result = collections.OrderedDict()
        for i,_rptpl in roi_points_dct.items():
            select_ans = []
            for j,_rp in enumerate(_rptpl):
                mask_roi = self.get_roi(_rp)
                total_pixels = cv2.countNonZero(mask_roi)
                if total_pixels > round(std_pixels/3.0*2):
                    select_ans.append(ans_options.get(j))

            ans_result[i] = select_ans
        mylog(ans_result=dict(ans_result))
        #############################
        #考号识别
        kh_points = sorted(self.kh_points,key=lambda x:x["y"])
        mylog(std_x_points=self.std_x_points)
        std_point_x = self.std_x_points[0].get("x")
        std_point_w = self.std_x_points[0].get("w")

        avg_width = self.get_avg_width(kh_points)
        std_kh_x = std_point_x - avg_width*1.5*10
        print std_kh_x,99999999999999999999999999
        for _kh in kh_points:
            _kh_x = _kh.get("x")
            kh_num = abs(round((_kh_x - std_kh_x)/(1.5*avg_width)))
            print kh_num

    def get_avg_width(self,cnts):
        """
        """
        w_list = []
        for _cnt in cnts:
            w_list.append(_cnt.get("w"))
        avg_width = sum(w_list)/len(w_list)
        return avg_width



if __name__ == "__main__":
    img_path = "test.jpg"
    #quenos = [1,4,7,8,9,10,11] 
    #quenos = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25] 
    #quenos = [1,2,3,4,5,6,7,8,9,10,11] 
    quenos = [1,2,3,4,5,6,7,8,9,10,11,12] 
    #quenos = [2,3,4,5] 
    #quenos = [1,2,3,4,5,6,7,8,9] 

    dpobj = DetectPaper(img_path,quenos)
    dpobj.rec_all_fill_cnts()
    dpobj.rec_paper()

