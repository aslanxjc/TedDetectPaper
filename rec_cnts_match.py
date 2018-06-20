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

class CntsFilter:
    def __init__(self,cnts=None,kh_std_dct={}):
        """
        """
        self.cnts = cnts
        self.kh_std_dct = kh_std_dct

    def denoise_data_bys(self,lst=[]):
        """算法面积去噪声
        """
        tmp = []

        lst = sorted(lst,key=lambda x:x["w"],reverse=True)
        mid_dct = lst[len(lst)/2]
        mid_s = mid_dct.get("w")*mid_dct.get("h")
        for _dct in lst:
            w = _dct.get("w")
            h = _dct.get("h")
            if w < mid_dct.get("w")/3:
                continue
            if h < mid_dct.get("h")/3:
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

        return (kh_cnts_lst,ans_cnts_lst)


    def get_kh_cnts_lst(self):
        """
        """
        kh_cnts_lst,_ = self.sep_kh_ans()
        #去噪声
        kh_cnts_lst = denoise_data_bys(kh_cnts_lst)
        return kh_cnts_lst


    def get_ans_cnts_lst(self):
        """
        """
        _,ans_cnts_lst = self.sep_kh_ans()
        #去噪声
        ans_cnts_lst,_ = denoise_data_bys(ans_cnts_lst)
        return ans_cnts_lst



def main():
    cntsfilter = CntsFilter()
    pass



if __name__ == "__main__":
    main()
