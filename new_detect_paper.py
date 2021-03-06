#-*-coding:utf-8 -*-
import os
import sys
import cv2
import collections
from collections import OrderedDict
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
        all_cnts,std_kh_line = get_ans_area_cnts(self.inverse_image_path)
        #答题框高度
        #ah = self.org_cut_point[3]-self.org_cut_point[1]
        #mylog(all_cnts=all_cnts)
        return all_cnts,std_kh_line

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

    
    def get_kh_roi(self,std_kh_line,rline,std_point_w):
        """
        """
        x1 = std_kh_line.get("x1")
        y1 = std_kh_line.get("y1")
        x2 = std_kh_line.get("x2")
        y2 = std_kh_line.get("y2")

        print y2-y1,888888888888
        std_h = round(abs(y2-y1)/6.0)
        print std_h,9999999999999999999
        #原来的
        std_w = abs(rline.get("x1")-std_kh_line.get("x1"))/10
        #std_w = abs(rline.get("x1")-(std_kh_line.get("x1")+std_point_w*2))/10

        kh_roi_dct = collections.OrderedDict()
        mask = np.zeros(self.thresh.shape[0:2],dtype=np.uint8) 
        for i,_ny in enumerate(range(0,6)):
            kh_nums = []
            for j,nx in enumerate(range(0,10)):
                #原来的
                _roi_point = (int(x1+std_w*1*j+2),int(y1+i*std_h),int(x1+(j+1)*std_w*1),int(y1+(i+1)*std_h+2))

                #if j==0:
                #    _roi_point = (int(x1+std_w*j+std_w*0.1+2),int(y1+i*std_h),int(x1+(j+1)*std_w+std_w*0.1),int(y1+(i+1)*std_h+2))
                ##elif j==9:
                ##    _roi_point = (int(x1+std_w*j+2),int(y1+i*std_h),int(x1+(j+1)*std_w),int(y1+(i+1)*std_h+2))
                #else:
                #    _roi_point = (int(x1+std_w*1*j+2),int(y1+i*std_h),int(x1+(j+1)*std_w*1),int(y1+(i+1)*std_h+2))

                kh_nums.append(_roi_point)

                cv2.rectangle(mask,(_roi_point[0],_roi_point[1]),
                    (_roi_point[2],_roi_point[3]),255,-1)
                mask = cv2.bitwise_and(self.thresh, self.thresh, mask=mask)
                #cv2.drawContours(mask, [c], -1, 255, -1)
                #break
            #break
            kh_roi_dct[i] = kh_nums

        #输出二值化操作后的图像
        spt_lst = os.path.splitext(self.inverse_image_path)
        close_path = spt_lst[0] + '_thresh_mask_kh' + spt_lst[1]
        cv2.imwrite(close_path,mask)

        return kh_roi_dct


    def filter_lines(self,lines_data,std_point_x):
        """
        """
        new_lines_data = []
        new_lines_data_pre = []

        std_point_x_x = std_point_x.get("x")
        print std_point_x_x,777777777777777777
        std_point_x_w = std_point_x.get("w")
        std_point_x_h = std_point_x.get("h")
        #答题框高度
        ah = self.org_cut_point[2][1]-self.org_cut_point[1][1]
        for _ld in lines_data:
            #if _ld.get("h")<ah/4*3:
            if _ld.get("h")<std_point_x_h*10:
                continue
            new_lines_data_pre.append(_ld)
            if abs(std_point_x_x-_ld.get("x2"))>std_point_x_w*3:
                continue
            new_lines_data.append(_ld)

        mylog(new_lines_data_pre=new_lines_data_pre)

        #考号标准线识别
        lines_data = sorted(new_lines_data,key=lambda x:x["x1"])

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

        _lines = []
        for x,_lines in tmp_dct.items():
            _lines = sorted(_lines,key=lambda x:x["h"],reverse=True)
            _lines = _lines[0]
            #return _lines[0]

        #向前寻找标准线
        new_lines_data_pre = filter(lambda x:x["x1"]<_lines["x1"],new_lines_data_pre)
        
        #前面标线的最大值
        pre_max_h = sorted(new_lines_data_pre,key=lambda x:x["h"],reverse=True)[0]["h"]

        #按x倒序排
        _std_lines = None
        new_lines_data_pre = sorted(new_lines_data_pre,key=lambda x:x["x1"],reverse=True)
        #mylog(new_lines_data_pre=new_lines_data_pre)
        for _l in new_lines_data_pre:
            if abs(_l.get("h")-pre_max_h) < std_point_x_w/3\
                     and abs(_l.get("x1")-_lines["x1"]) > std_point_x_w*10:
                _std_lines = _l
                break
        return _std_lines,_lines

    def filter_lines_en(self,lines_data,std_point_x_l):
        """上下结构标准线寻找
        """
        new_lines_data = []
        new_lines_data_pre = []

        std_point_x_l_x = std_point_x_l.get("x")
        std_point_x_l_w = std_point_x_l.get("w")
        #答题框高度
        ah = self.org_cut_point[2][1]-self.org_cut_point[1][1]
        for _ld in lines_data:
            if _ld.get("h")<ah/4:
                continue
            if _ld.get("h")>ah/3*2:
                new_lines_data.append(_ld)
                continue
            if _ld.get("x1")>std_point_x_l_x:
                continue

            new_lines_data_pre.append(_ld)

        #右侧基准标线
        new_lines_data = sorted(new_lines_data,key=lambda x:x["x1"])
        mylog(new_lines_data=new_lines_data)
        mylog(new_lines_data_pre=new_lines_data_pre)
        _lines = new_lines_data[0]


        #向前寻找标准线
        new_lines_data_pre = filter(lambda x:x["x1"]<_lines["x1"],new_lines_data_pre)

        #前面标线的最大值
        pre_max_h = sorted(new_lines_data_pre,key=lambda x:x["h"],reverse=True)[0]["h"]

        #按x倒序排
        _std_lines = None
        new_lines_data_pre = sorted(new_lines_data_pre,key=lambda x:x["x1"],reverse=True)
        #mylog(new_lines_data_pre=new_lines_data_pre)
        for _l in new_lines_data_pre:
            if abs(_l.get("h")-pre_max_h) < std_point_x_l_w/3\
                     and abs(_l.get("x1")-_lines["x1"]) > std_point_x_l_w*10:
                _std_lines = _l
                break
        return _std_lines,_lines




    def rec_paper(self,flag = False):
        """
        """
        all_cnts,std_kh_lines = self.rec_all_fill_cnts()
        cntsfilter = CntsFilter(all_cnts,self.quenos,self.org_cut_point)
        self.std_x_points,self.std_y_points,self.kh_points = cntsfilter.sep_std_cnts(flag)
        #mylog(std_x_points=self.std_x_points)
        #mylog(std_y_points=self.std_y_points)
        #mylog(kh_points=self.kh_points)
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
        #mylog(std_x_points=self.std_x_points)
        std_point_x = self.std_x_points[0].get("x")
        std_point_w = self.std_x_points[0].get("w")

        if flag:
            std_kh_line,rline = self.filter_lines_en(std_kh_lines,self.std_x_points[-1])
        else:
            std_kh_line,rline = self.filter_lines(std_kh_lines,self.std_x_points[0])

        #print std_kh_line,rline,888888888888888888888888888

        avg_width = self.get_avg_width(kh_points)
        std_kh_x = std_point_x - avg_width*1.5*10
        kh_result = collections.OrderedDict()
        if std_kh_line:
            kh_roi_dct = self.get_kh_roi(std_kh_line,rline,std_point_w)
            for i,_kr in kh_roi_dct.items():
                kh = []
                for j,_rp in enumerate(_kr):
                    _tmp = {}
                    mask_roi = self.get_roi(_rp)
                    total_pixels = cv2.countNonZero(mask_roi)
                    _tmp["n"] = j
                    _tmp["v"] = total_pixels
                    kh.append(_tmp)
                kh = sorted(kh,key=lambda x:x["v"],reverse=True)

                kh_result[i] = kh[0].get("n")
            mylog(kh_result=dict(kh_result))


        #for _kh in kh_points:
        #    _kh_x = _kh.get("x")
        #    print _kh_x - std_kh_x
        #    print (_kh_x - std_kh_x)/(1.5*avg_width)
        #    kh_num = abs(int((_kh_x - std_kh_x)/(1.5*avg_width)))
        #    print kh_num
        return kh_result,ans_result

    def get_avg_width(self,cnts):
        """
        """
        cnts = sorted(cnts,key=lambda x:x["w"])
        w_list = []
        for _cnt in cnts:
            w_list.append(_cnt.get("w"))
        avg_width = sum(w_list)/len(w_list)
        #return avg_width
        return cnts[-1].get("w")



if __name__ == "__main__":
    img_path = "test.jpg"
    #quenos = [1,4,7,8,9,10,11] 
    #quenos = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25] 
    quenos = [2,3,4,5] 
    #quenos = [1,2,3,4,5,6,7,8,9] 
    quenos = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18] 
    #quenos = [1,2,3,4,5,6,7,8,9,10,11,12,13] 
    quenos = [1,2,3,4,5,6,7,8,9,10,11] 
    quenos = [1,4,7,8,9,10,11] 
    quenos = [1,2,3,5,6,7,8,9,10,11,12,13,14] 
    #quenos = [1,2,3,5,6,8,9,10,11,12,13] 
    quenos = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40] 
    #quenos = [1,2,3,4,5,6] 
    quenos = [1,2,3,4,5,6,7,8,9,10,11,12,13] 
    quenos = [1,2,3,4,5,6,7,8,9,10] 

    dpobj = DetectPaper(img_path,quenos)
    dpobj.rec_all_fill_cnts()
    dpobj.rec_paper()

