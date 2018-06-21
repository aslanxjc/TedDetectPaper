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
from collections import OrderedDict

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
    def __init__(self,cnts_data=[],quenos=[],org_ans_point=()):
        """
        """
        self.cnts_data = cnts_data
        self.org_ans_point = org_ans_point
        self.quenos = quenos

    def get_std_wh(self):
        """
        """
        wh_cnts = sorted(self.cnts_data,key=lambda x:x["w"])
        std_w = wh_cnts[len(wh_cnts)/2].get("w")
        std_h = wh_cnts[len(wh_cnts)/2].get("h")
        #mylog(wh_cnts=wh_cnts)
        return std_w,std_h

    def denoise_data_bwh(self,cnts=[]):
        """pass
        """
        _cnts = []
        std_w,std_h = self.get_std_wh()
        for _cnt in cnts:
            w = _cnt.get("w")
            h = _cnt.get("h")
            if w < std_w/2.0:
                continue
            if h < std_h/2.0:
                continue
            _cnts.append(_cnt)
        return _cnts

    def group_by_y(self,cnts_data=[],std_queno=[]):
        """
        """
        std_w,std_h = self.get_std_wh()
        tmp_dct = OrderedDict()
        for _point in cnts_data: 
            x = _point.get("x")
            y = _point.get("y")
            w = _point.get("w")
            h = _point.get("h")
            if w < std_w/2.0:
                continue
            if h < std_h/2.0:
                continue

            if not tmp_dct:
                tmp_dct[y] = [_point]
            else:
                if (y-tmp_dct.keys()[-1]) < std_h/2:
                    tmp_dct[tmp_dct.keys()[-1]].append(_point)
                else:
                    tmp_dct[y] = [_point]
        #mylog(tmp_dct=dict(tmp_dct))
        #字典反序
        _keys = tmp_dct.keys()
        _keys.reverse()
        for _key in _keys:
            _std_x_cnts = tmp_dct[_key]
            #if len(_std_x_cnts) == 7:
            if len(_std_x_cnts) == len(self.quenos):
                std_x_points = _std_x_cnts
                return std_x_points
        return None

    def group_by_x(self,cnts_data=[],std_queno=[]):
        """
        """
        std_w,std_h = self.get_std_wh()
        tmp_dct = OrderedDict()
        for _point in cnts_data: 
            x = _point.get("x")
            y = _point.get("y")
            w = _point.get("w")
            h = _point.get("h")
            if w < std_w/2.0:
                continue
            if h < std_h/2.0:
                continue

            if not tmp_dct:
                tmp_dct[x] = [_point]
            else:
                if (x-tmp_dct.keys()[-1]) < std_w/4:
                    tmp_dct[tmp_dct.keys()[-1]].append(_point)
                else:
                    tmp_dct[x] = [_point]
        #mylog(tmp_dct=dict(tmp_dct))
        #字典反序
        _keys = tmp_dct.keys()
        _keys.reverse()
        for _key in _keys:
            _std_y_cnts = tmp_dct[_key]
            if len(_std_y_cnts) >= 4:
                std_y_points = _std_y_cnts
                return std_y_points
        return None

    def sep_std_cnts(self,flag=False):
        """
        """
        std_w,std_h = self.get_std_wh()
        print std_w,11111111111111
        print std_h,22222222222222
        self.cnts_data = self.denoise_data_bwh(self.cnts_data)
        #按y进行排序
        #print self.cnts_data,3333333333333
        _ycnts = sorted(self.cnts_data,key=lambda x:x["y"])
        std_x_points = self.group_by_y(_ycnts)
        std_x_points = sorted(std_x_points,key=lambda x:x["x"])
        mylog(std_x_points=std_x_points)
        #按x进行排序
        _xcnts = sorted(self.cnts_data,key=lambda x:x["x"])
        std_y_points = self.group_by_x(_xcnts)
        std_y_points = sorted(std_y_points,key=lambda x:x["y"])
        mylog(std_y_points=std_y_points)

        #答题填涂的轮廓
        ans_points = filter(lambda x:x["x"]>std_x_points[0]["x"]-std_w \
                        and x["y"]>std_y_points[0]["y"]-std_h,self.cnts_data)

        if flag:
            #上下结构(英语)考号区域轮廓
            kh_points = filter(lambda x:x["y"]<std_y_points[0]["y"]-std_h,self.cnts_data)
            mylog(kh_points=kh_points)
        else:
            #考号区域轮廓
            kh_points = filter(lambda x:x["x"]<std_x_points[0]["x"]-std_w,self.cnts_data)
            mylog(kh_points=kh_points)

        
        for _p in std_x_points:
            try:
                ans_points.remove(_p)
            except:
                pass
        for _p in std_y_points:
            try:
                ans_points.remove(_p)
            except:
                pass
        #mylog(ans_points=ans_points)

        return std_x_points,std_y_points,kh_points

if __name__ == "__main__":
    pass
