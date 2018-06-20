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

from rec_closed_cnts import (get_kh_std_dct,get_ans_area_cnts,
                                get_en_kh_std_dct,get_en_ans_area_cnts)


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

class CntsFilter:
    def __init__(self,path="test_cut.jpg",std_point=None,std_quenos=[]):
        """
        """
        self.cnts = get_ans_area_cnts(path,std_point,std_quenos=std_quenos)
        self.kh_std_dct = get_kh_std_dct(path)

    def denoise_data_bys(self,lst=[]):
        """算法面积去噪声
        """
        tmp = []

        lst = sorted(lst,key=lambda x:x["w"],reverse=True)
        mid_dct = lst[len(lst)/2]
        mid_s = mid_dct.get("w")*mid_dct.get("h")
        mid_w = mid_dct.get("w")
        mid_h = mid_dct.get("h")
        mid_s = mid_dct.get("w")*mid_dct.get("h")
        #mid_s = pow(mid_w,mid_w)*pow(mid_h,mid_h)
        for _dct in lst:
            x = _dct.get("x")
            if not x:
                continue
            w = _dct.get("w")
            h = _dct.get("h")
            #if w < mid_dct.get("w")/3.0*2:
            #    continue
            if w > mid_w*2: 
                continue
            if h > mid_h*2: 
                continue
            #if h < mid_h/3.0*2.0:
            #    print 3333333333333
            #    continue
            s = w*h
            #s = pow(w,w)*pow(h,h)
            if s > mid_s/2:
            #if s > mid_s/3.0*2:
                tmp.append(_dct)

        return tmp


    def sep_kh_ans(self,kh_std_dct={}):
        """
        """
        kh_cnts_lst = []
        ans_cnts_lst = []

        #考号基准分割线
        kh_std_x = self.kh_std_dct.get('x')
        kh_std_w = self.kh_std_dct.get('w')

        kh_max_x = kh_std_x + kh_std_w 

        fill_cnts_lst = sorted(self.cnts,key=lambda x:x['x'])

        for _cdct in fill_cnts_lst:
            _x = _cdct.get('x')
            _y = _cdct.get('y')
            _w = _cdct.get('w')

            if not _x:
                continue
            if _cdct.get('x') < kh_std_x:
                continue

            if _cdct.get('x') < kh_max_x:
                kh_cnts_lst.append(_cdct)
            else:
                ans_cnts_lst.append(_cdct)

        mylog(kh_cnts_lst=kh_cnts_lst)

        return (kh_cnts_lst,ans_cnts_lst)


    def get_kh_cnts_lst(self):
        """
        """
        kh_cnts_lst,_ = self.sep_kh_ans()
        #去噪声
        kh_cnts_lst = self.denoise_data_bys(kh_cnts_lst)
        mylog(kh_cnts_lst=kh_cnts_lst)
        return kh_cnts_lst


    def get_ans_cnts_lst(self):
        """
        """
        _,ans_cnts_lst = self.sep_kh_ans()
        #去噪声
        ans_cnts_lst = self.denoise_data_bys(ans_cnts_lst)
        return ans_cnts_lst


###########################################################
#英语上下结构
class EnCntsFilter:
    def __init__(self,path="test_cut.jpg",std_point=None,std_quenos=[]):
        """
        """
        self.cnts = get_en_ans_area_cnts(path,std_point,std_quenos=std_quenos)
        self.kh_std_dct = get_en_kh_std_dct(path)

    def denoise_data_bys(self,lst=[]):
        """算法面积去噪声
        """
        tmp = []

        lst = sorted(lst,key=lambda x:x["w"],reverse=True)
        mid_dct = lst[len(lst)/2]
        mid_s = mid_dct.get("w")*mid_dct.get("h")
        mid_w = mid_dct.get("w")
        mid_h = mid_dct.get("h")
        mid_s = mid_dct.get("w")*mid_dct.get("h")
        #mid_s = pow(mid_w,mid_w)*pow(mid_h,mid_h)
        for _dct in lst:
            x = _dct.get("x")
            if not x:
                continue
            w = _dct.get("w")
            h = _dct.get("h")
            if w > mid_w*2: 
                continue
            if h > mid_h*2: 
                continue
            s = w*h
            if s > mid_s/2:
                tmp.append(_dct)

        return tmp


    def sep_kh_ans(self,kh_std_dct={}):
        """
        """
        kh_cnts_lst = []
        ans_cnts_lst = []

        #考号基准分割线
        kh_std_x = self.kh_std_dct.get('x')
        kh_std_w = self.kh_std_dct.get('w')
        kh_std_y = self.kh_std_dct.get('y')

        kh_max_x = kh_std_x + kh_std_w 

        #fill_cnts_lst = sorted(self.cnts,key=lambda x:x['x'])
        fill_cnts_lst = sorted(self.cnts,key=lambda x:x['y'])

        for _cdct in fill_cnts_lst:
            _x = _cdct.get('x')
            _y = _cdct.get('y')
            _w = _cdct.get('w')

            if _x < kh_std_x and _y < kh_std_y:
                continue

            if _y < kh_std_y:
                kh_cnts_lst.append(_cdct)
            else:
                ans_cnts_lst.append(_cdct)

            #if _cdct.get('x') < kh_max_x:
            #    kh_cnts_lst.append(_cdct)
            #else:
            #    ans_cnts_lst.append(_cdct)

        mylog(kh_cnts_lst=kh_cnts_lst)

        return (kh_cnts_lst,ans_cnts_lst)


    def get_kh_cnts_lst(self):
        """
        """
        kh_cnts_lst,_ = self.sep_kh_ans()
        #去噪声
        kh_cnts_lst = self.denoise_data_bys(kh_cnts_lst)
        mylog(kh_cnts_lst=kh_cnts_lst)
        return kh_cnts_lst


    def get_ans_cnts_lst(self):
        """
        """
        _,ans_cnts_lst = self.sep_kh_ans()
        #去噪声
        ans_cnts_lst = self.denoise_data_bys(ans_cnts_lst)
        return ans_cnts_lst






def main():
    std_point = {'y': 102, 'x': 174, 'w': 25, 'h': 32}
    std_quenos = [(1, 0), (2, 0), (3, 0), (4, 0),
                     (1, 0), (2, 0), (3, 0), (4, 0), (5, 0),
                     (6, 0), (7, 0), (8, 0), (9, 0), (10, 0),
                     (11, 0), (12, 0), (13, 0), (14, 0), (15, 0),
                     (16, 0), (17, 0), (18, 0), (19, 0), (20, 0),
                     ]
    #cntsfilter = CntsFilter()
    #kh_cnts_lst = cntsfilter.get_kh_cnts_lst()
    #mylog(kh_cnts_lst=kh_cnts_lst)
    #ans_cnts_lst = cntsfilter.get_ans_cnts_lst()
    #mylog(ans_cnts_lst=ans_cnts_lst)

    #################################
    #英语上下结构
    cntsfilter = EnCntsFilter("test_cut.jpg",std_point,std_quenos)
    kh_cnts_lst = cntsfilter.get_kh_cnts_lst()
    mylog(kh_cnts_lst=kh_cnts_lst)
    ans_cnts_lst = cntsfilter.get_ans_cnts_lst()
    mylog(ans_cnts_lst=ans_cnts_lst)
    pass



if __name__ == "__main__":
    main()
