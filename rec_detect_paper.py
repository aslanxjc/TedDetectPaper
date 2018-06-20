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


import os
import sys
import cv2
import collections

sys.path.append('.')
os.environ.setdefault("DJANGO_SETTINGS_MODULE","ss_mark.settings")

from ss_mark.models import Paper as ssPaper,PaperPieces
from django.db.models import Q
#from utils.DetectPaper.rec_ans_area import RecAnsArea
from utils.DetectPaper.new_rec_ans_area import RecAnsArea


#from rec_closed_cnts import get_kh_std_dct,get_ans_area_cnts
#from rec_cnts_filter import CntsFilter
#from detect_image_std import get_std_point 

from utils.DetectPaper.rec_closed_cnts import (get_kh_std_dct,get_ans_area_cnts,
                                        get_en_kh_std_dct,get_en_ans_area_cnts)
from utils.DetectPaper.rec_cnts_filter import CntsFilter,EnCntsFilter
from utils.DetectPaper.detect_image_std import get_std_point 
from common.common import *

global std_quenos


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

class DetectPaper:
    def __init__(self,path="test_cut.jpg",std_point=None,
                    std_quenos=[],ans_th_rate=[],ans_xx_rate=[]):
        """
        """
        #cv2图片对象
        self.image = cv2.imread(path)
        #图片的宽高度
        self.width = self.image.shape[1]
        self.height = self.image.shape[0]

        self.cntsfilter = CntsFilter(path,std_point,std_quenos)
        #self.cnts = cnts
        self.kh_std_dct = get_kh_std_dct(path)
        self.std_quenos=std_quenos
        self.ans_th_rate = ans_th_rate
        self.ans_xx_rate = ans_xx_rate

    def cal_kaohao_str(self):
        """
        """
        kh_cnts_lst = self.cntsfilter.get_kh_cnts_lst()

        #考号基准分割线
        kh_std_x = self.kh_std_dct.get('x')
        kh_std_w = self.kh_std_dct.get('w')

        kh_max_x = kh_std_x + kh_std_w 

        kh_max_x = kh_std_x


        #卡号填图区域的平均宽度
        avg_w = sum([x['w'] for x in kh_cnts_lst])/len(kh_cnts_lst) \
                if kh_cnts_lst else 0
        #计算基准大小
        meta_w = (kh_std_w - avg_w*10)/11

        #考号轮廓按Y排序
        kh_cnts_lst = sorted(kh_cnts_lst,key=lambda x:x['y'])

        kh_lst = []
        for _c_x in kh_cnts_lst:
            w = _c_x.get('w')
            dis_x = _c_x.get('x') - kh_std_x

            kh_no = dis_x/(avg_w+meta_w)
            kh_no = int(round(kh_no))

            kh_lst.append(kh_no)

        #mylog(kh_lst=kh_lst)

        kh_str = ''.join([str(x) for x in kh_lst])

        return kh_str


    def get_mid_repeat(self,lst=[]):
        """
        """
        ws = [x["w"] for x in lst]
        if len(ws) > len(list(set(ws))):
            #取众数
            #_w = max(ws.count(x) for x in set(ws))
            import numpy as np
            a = np.array(ws)
            counts = np.bincount(a)
            _w = np.argmax(counts)
        else:
            _w = ws[len(ws)/2] 

        return _w
    



    def cal_ans_lst(self):
        """通过轮廓计算出答案
        """
        ans_cnts_lst = self.cntsfilter.get_ans_cnts_lst()

        self.std_quenos.sort()

        char_lst = ["A","B","C","D","E","F","G","H","I"]

        #计算对比比例
        #th_rate_dis = (ans_th_rate[-1]-ans_th_rate[0])/len(ans_th_rate)
        th_rate_dis = (self.ans_th_rate[-1]-self.ans_th_rate[0])/len(self.std_quenos)
        th_rate_dis = round(th_rate_dis,5)

        xx_rate_dis = (self.ans_xx_rate[-1] - self.ans_xx_rate[0])/len(char_lst)
        xx_rate_dis = round(xx_rate_dis,5)

        #设置标准题号坑
        std_ans_queno = []
        for i,qn in enumerate(self.std_quenos):
            dct = {}
            dct['queno'] = str(self.std_quenos[i])
            dct['rate'] = self.ans_th_rate[i]
            dct['cnts'] = []
            dct['range'] = self.width*self.ans_th_rate[i]
            std_ans_queno.append(dct)

        mylog(std_ans_queno=std_ans_queno)


        #是指标准选项标靶
        std_ans_xx = []
        for i,xx in enumerate(self.ans_xx_rate):
            dct = {}
            dct['rate'] = self.ans_xx_rate[i] 
            dct["xx"] = char_lst[i] 
            std_ans_xx.append(dct)

        #相邻X间距小于1/2最小宽度最归结为一组
        #取出轮廓里面的最小宽度
        mylog(ans_cnts_lst=ans_cnts_lst)
        ans_cnts_lst = sorted(ans_cnts_lst,key=lambda x:x["x"])
        mylog(ans_cnts_lst=ans_cnts_lst)
        w_lst = [x['w'] for x in ans_cnts_lst]
        w_lst.sort()
        min_w = w_lst[0]

        import collections
        julei = []
        dct = collections.OrderedDict()
        #ans_cnts_lst.append({'h':0,'w':0,'x':0,'y':0})
        for _a in ans_cnts_lst:
            w = _a.get('w')
            x = _a.get('x')
            #print dct.keys(),6666666666666666666666666
            k = dct.keys()[-1] if dct.keys() else 0

            if x-k < min_w/2:
            #if x-k < w/2:
                dct[k].append(_a)
            else:
                dct[x] = [_a]

        for k,v in dct.items():
            _dct = {}
            _dct['x'] = k
            _dct['w'] = v[0].get('w') if v else 0
            _dct['cnts'] = v
            julei.append(_dct)

        mylog(julei=julei)
        for i,std_que in enumerate(std_ans_queno):
            #第i题的标准比率
            th_rate = std_que.get('rate')
            try:

                for j,_jl in enumerate(julei):
                    _cnts = julei[j].get('cnts')

                    x = julei[j].get('x')

                    w = julei[j].get('w')

                    w = self.get_mid_repeat(julei)

                    this_rate = float(x)/self.width
                    _jl["this_rate"] = this_rate

                    #原来的范围
                    #if len(julei) >= 20:
                    #    x = x+w/2
                    #else:
                    #    x = x+w/3.5
                    x = x+w/5
                    #x = x+w/3

                    std_ans_queno[i]['this_rate'] = this_rate
                    std_ans_queno[i]['this_x'] = this_rate/th_rate

                    if len(julei) >= 20:
                        #_std_th_range = [self.width*th_rate-w/2,self.width*th_rate+w/2+10]
                        _std_th_range = [self.width*th_rate-w/2,self.width*th_rate+w/2]
                    else:
                        #_std_th_range = [self.width*th_rate-w/2,self.width*th_rate+w/2+10]
                        _std_th_range = [self.width*th_rate-w/2,self.width*th_rate+w/2]

                    print _std_th_range,11111111111111111111
                    if x>=_std_th_range[0] and x<=_std_th_range[1]:
                        std_ans_queno[i]['cnts'].extend(_cnts)
                        julei.remove(_jl)
            except Exception ,e:
                import traceback
                traceback.print_exc()
        #mylog(std_ans_queno=std_ans_queno)
        #print 7777777777777777777777777777777

        ################################################
        #将匹配好的题号与轮廓进行处理,解析出选择的答案到底是A,B,C,D
        for que_cnt_dct  in std_ans_queno:

            queno = que_cnt_dct.get('queno')

            que_cnt_dct['ans'] = []
            _cnts = que_cnt_dct.get('cnts')
            #如果有则按Y进行排序
            if _cnts:
                _cnts = sorted(_cnts,key=lambda x:x['y'])

                #将每个题号下的轮廓循环与标准选项比例做比较
                #选项标准比率
                #上面空白处的距离
                dis_y =  self.ans_xx_rate[0]*self.height


                #选项高度
                xx_dis = self.height-dis_y
                #计算平均高度
                avg_h = sum(x['h'] for x in _cnts)/len(_cnts)
                #计算间隙的高度
                meta_h = (xx_dis - avg_h*len(self.ans_xx_rate))/(len(self.ans_xx_rate)-1)

                #print u'题号:'
                #print que_cnt_dct['queno']

                for c in _cnts:
                    ##通过距离在范围内计算
                    y = c.get('y')
                    h = c.get('h')

                    y = y+avg_h/4

                    std_y_ranges = [[self.height*x-avg_h/1.2,self.height*x+avg_h/1.2] for x in self.ans_xx_rate]
                    std_y_ranges = [[self.height*x-avg_h/1.1,self.height*x+avg_h/1.1] for x in self.ans_xx_rate]
                    
                    flag = True
                    for i,yr in enumerate(std_y_ranges):
                        if y>=yr[0] and y<=yr[1]:
                            que_cnt_dct['ans'].append(char_lst[i])
                            flag = False
                    if flag:
                        que_cnt_dct['ans'].append("xx")

            else:
                #如果没有则该题未填图,人为设置为x
                que_cnt_dct["ans"] = ["x"]
        ################################################

        #返回识别出的答案
        shit_lst = []
        for _shit_dct  in std_ans_queno:
            _tmp_dct = {}
            _tmp_dct['queno'] = _shit_dct.get('queno')
            _tmp_dct["ans"] = _shit_dct.get("ans")
            shit_lst.append(_tmp_dct)
        mylog(shit_lst=shit_lst)
        return shit_lst



#############################################################################
#英语上下结构
class DetectEnPaper:
    def __init__(self,path="test_cut.jpg",std_point=None,
                    std_quenos=[],ans_th_rate=[],ans_xx_rate=[]):
        """
        """
        #cv2图片对象
        self.image = cv2.imread(path)
        #图片的宽高度
        self.width = self.image.shape[1]
        self.height = self.image.shape[0]

        self.cntsfilter = EnCntsFilter(path,std_point,std_quenos)
        #self.cnts = cnts
        self.kh_std_dct = get_en_kh_std_dct(path)
        self.std_quenos=std_quenos
        self.ans_th_rate = ans_th_rate
        self.ans_xx_rate = ans_xx_rate

    def cal_kaohao_str(self):
        """
        """
        kh_cnts_lst = self.cntsfilter.get_kh_cnts_lst()

        #考号基准分割线
        kh_std_x = self.kh_std_dct.get('x')
        kh_std_w = self.kh_std_dct.get('w')

        kh_max_x = kh_std_x + kh_std_w 

        kh_max_x = kh_std_x


        #卡号填图区域的平均宽度
        avg_w = sum([x['w'] for x in kh_cnts_lst])/len(kh_cnts_lst) \
                if kh_cnts_lst else 0
        #计算基准大小
        meta_w = (kh_std_w - avg_w*10)/11

        #考号轮廓按Y排序
        kh_cnts_lst = sorted(kh_cnts_lst,key=lambda x:x['y'])

        kh_lst = []
        for _c_x in kh_cnts_lst:
            w = _c_x.get('w')
            dis_x = _c_x.get('x') - kh_std_x

            kh_no = dis_x/(avg_w+meta_w)
            kh_no = int(round(kh_no))

            kh_lst.append(kh_no)

        mylog(kh_lst=kh_lst)

        kh_str = ''.join([str(x) for x in kh_lst])

        return kh_str


    def get_mid_repeat(self,lst=[]):
        """
        """
        ws = [x["w"] for x in lst]
        if len(ws) > len(list(set(ws))):
            #取众数
            #_w = max(ws.count(x) for x in set(ws))
            import numpy as np
            a = np.array(ws)
            counts = np.bincount(a)
            _w = np.argmax(counts)
        else:
            _w = ws[len(ws)/2] 

        return _w
    

    def cut_ans_area(self,cut_point=None):
        '''
        从试卷裁剪出答题卡区域图片
        '''
        img = Image.open(self.path)
        region = img.crop(cut_point)

        spt_lst = os.path.splitext(self.path)
        cut_path = spt_lst[0] + '_cut_'+str(cut_point)+ spt_lst[1]

        region.save(cut_path)

        return cut_path


    def get_std_xy(self):
        """
        """
        ans_cnts_lst = self.cntsfilter.get_ans_cnts_lst()
        #x方向
        #轮廓按X排序
        ans_cnts_lst = sorted(ans_cnts_lst,key=lambda x:x['x'])
        first_cnt = ans_cnts_lst[0]
        first_x = first_cnt.get('x') 

        #题号
        x_lst = []
        #选项
        y_lst = []
        for _c in ans_cnts_lst:
            _x = _c.get('x')
            if _x-first_x < 10:
                y_lst.append(_c)
            else:
                x_lst.append(_c)

        x_lst.insert(0,first_cnt)

        x_rate = []
        for _c in x_lst:
            th_x = _c.get('x')
            th_w = _c.get('w')
            #x_rate.append((th_x+th_w/10)/float(self.width))
            x_rate.append((th_x+th_w/2)/float(self.width))

        y_rate = []
        for _c in y_lst:
            th_y = _c.get('y')
            th_h = _c.get('h')
            y_rate.append((th_y+th_h/2)/float(self.height))

        return x_rate,y_rate



    def cal_ans_lst(self):
        """通过轮廓计算出答案
        """
        kh_std_h = self.kh_std_dct.get('h')
        #cut_point = (0,kh_std_h,self.width,self.height
        #ans_cut_path = self.cut_ans_area(cut_point)
        #self.height = self.height-kh_std_h

        #print self.height,9999999999999999999999
        #raise Exception

        ans_cnts_lst = self.cntsfilter.get_ans_cnts_lst()

        self.std_quenos.sort()

        char_lst = ["A","B","C","D","E","F","G","H","I"]

        #计算对比比例
        #th_rate_dis = (ans_th_rate[-1]-ans_th_rate[0])/len(ans_th_rate)
        th_rate_dis = (self.ans_th_rate[-1]-self.ans_th_rate[0])/len(self.std_quenos)
        th_rate_dis = round(th_rate_dis,5)

        xx_rate_dis = (self.ans_xx_rate[-1] - self.ans_xx_rate[0])/len(char_lst)
        xx_rate_dis = round(xx_rate_dis,5)

        #设置标准题号坑
        std_ans_queno = []
        for i,qn in enumerate(self.std_quenos):
            dct = {}
            dct['queno'] = str(self.std_quenos[i])
            dct['rate'] = self.ans_th_rate[i]
            dct['cnts'] = []
            dct['range'] = self.width*self.ans_th_rate[i]
            std_ans_queno.append(dct)

        mylog(std_ans_queno=std_ans_queno)


        #是指标准选项标靶
        std_ans_xx = []
        for i,xx in enumerate(self.ans_xx_rate):
            dct = {}
            dct['rate'] = self.ans_xx_rate[i] 
            dct["xx"] = char_lst[i] 
            std_ans_xx.append(dct)

        #相邻X间距小于1/2最小宽度最归结为一组
        #取出轮廓里面的最小宽度
        mylog(ans_cnts_lst=ans_cnts_lst)
        ans_cnts_lst = sorted(ans_cnts_lst,key=lambda x:x["x"])
        mylog(ans_cnts_lst=ans_cnts_lst)
        w_lst = [x['w'] for x in ans_cnts_lst]
        w_lst.sort()
        min_w = w_lst[0]

        import collections
        julei = []
        dct = collections.OrderedDict()
        #ans_cnts_lst.append({'h':0,'w':0,'x':0,'y':0})
        for _a in ans_cnts_lst:
            w = _a.get('w')
            x = _a.get('x')
            #print dct.keys(),6666666666666666666666666
            k = dct.keys()[-1] if dct.keys() else 0

            if x-k < min_w/2:
            #if x-k < w/2:
                dct[k].append(_a)
            else:
                dct[x] = [_a]

        for k,v in dct.items():
            _dct = {}
            _dct['x'] = k
            _dct['w'] = v[0].get('w') if v else 0
            _dct['cnts'] = v
            julei.append(_dct)

        mylog(julei=julei)
        for i,std_que in enumerate(std_ans_queno):
            #第i题的标准比率
            th_rate = std_que.get('rate')
            try:

                for j,_jl in enumerate(julei):
                    _cnts = julei[j].get('cnts')

                    x = julei[j].get('x')

                    w = julei[j].get('w')

                    w = self.get_mid_repeat(julei)

                    this_rate = float(x)/self.width
                    _jl["this_rate"] = this_rate

                    #原来的范围
                    #if len(julei) >= 20:
                    #    x = x+w/2
                    #else:
                    #    x = x+w/3.5
                    x = x+w/6
                    x = x+w/3

                    std_ans_queno[i]['this_rate'] = this_rate
                    std_ans_queno[i]['this_x'] = this_rate/th_rate

                    if len(julei) >= 20:
                        #_std_th_range = [self.width*th_rate-w/2,self.width*th_rate+w/2+10]
                        _std_th_range = [self.width*th_rate-w/2,self.width*th_rate+w/2]
                        _std_th_range = [self.width*th_rate-w/1.5,self.width*th_rate+w/1.5]
                    else:
                        #_std_th_range = [self.width*th_rate-w/2,self.width*th_rate+w/2+10]
                        _std_th_range = [self.width*th_rate-w/2,self.width*th_rate+w/2]

                    if x>=_std_th_range[0] and x<=_std_th_range[1]:
                        std_ans_queno[i]['cnts'].extend(_cnts)
                        julei.remove(_jl)
            except Exception ,e:
                import traceback
                traceback.print_exc()
        #mylog(std_ans_queno=std_ans_queno)
        #print 7777777777777777777777777777777

        ################################################
        #将匹配好的题号与轮廓进行处理,解析出选择的答案到底是A,B,C,D
        for que_cnt_dct  in std_ans_queno:

            queno = que_cnt_dct.get('queno')

            que_cnt_dct['ans'] = []
            _cnts = que_cnt_dct.get('cnts')
            #如果有则按Y进行排序
            if _cnts:
                _cnts = sorted(_cnts,key=lambda x:x['y'])

                #将每个题号下的轮廓循环与标准选项比例做比较
                #选项标准比率
                #上面空白处的距离
                dis_y =  self.ans_xx_rate[0]*self.height


                #选项高度
                xx_dis = self.height-dis_y
                #计算平均高度
                avg_h = sum(x['h'] for x in _cnts)/len(_cnts)
                #计算间隙的高度
                meta_h = (xx_dis - avg_h*len(self.ans_xx_rate))/(len(self.ans_xx_rate)-1)

                #print u'题号:'
                #print que_cnt_dct['queno']

                for c in _cnts:
                    ##通过距离在范围内计算
                    y = c.get('y')
                    h = c.get('h')

                    y = y+avg_h/4

                    #std_y_ranges = [[self.height*x-avg_h/1.2,self.height*x+avg_h/1.2] for x in self.ans_xx_rate]
                    std_y_ranges = [[self.height*x-avg_h/1.1,self.height*x+avg_h/1.1] for x in self.ans_xx_rate]
                    flag = True
                    for i,yr in enumerate(std_y_ranges):
                        #y = y-kh_std_h
                        if y>=yr[0] and y<=yr[1]:
                            que_cnt_dct['ans'].append(char_lst[i])
                            flag = False
                            break
                    if flag:
                        que_cnt_dct['ans'].append("xx")

            else:
                #如果没有则该题未填图,人为设置为x
                que_cnt_dct["ans"] = ["x"]
        ################################################

        #返回识别出的答案
        shit_lst = []
        for _shit_dct  in std_ans_queno:
            _tmp_dct = {}
            _tmp_dct['queno'] = _shit_dct.get('queno')
            _tmp_dct["ans"] = _shit_dct.get("ans")
            shit_lst.append(_tmp_dct)
        mylog(shit_lst=shit_lst)
        return shit_lst












def main():
    img_url = os.path.join(os.path.dirname(__file__),"test.jpg")
    paperno = "XZB2018LJDLB03006"
    paperno = "XZB2018RJWLX31001"
    paperno = "XZB2018LJDLB03003"
    paperno = "XZB2018RJWLX31002"
    paperno = "XZB2018RJWLB02001"
    paperno = "XZB2018RJWLB02002"
    paperno = "XZB2018RJYWB03001"
    paperno = "XZB2018LKHXX04001"
    paperno = "XZB2018RJYYB01001"
    paperno = "XZB2018RASXB03004"
    paperno = "XZB2018RJHXB02001"
    paperno = "XZB2018RJHXB02004"
    paperno = "XZB2018RJHXB02004"
    paperno = "XZB2018RJWLB02007"
    paperno = "XZB2018RJWLB02005"
    #std_point = {'y': 102, 'x': 174, 'w': 36, 'h': 49}
    #std_quenos = [(1, 0), (2, 0), (3, 0), (4, 0), (5, 0), (6, 0), (7, 0), (8, 0), (9, 0)]
    #ans_xx_rate = [0.35379061371841153, 0.5342960288808665, 0.7111913357400722, 0.8916967509025271]
    #ans_th_rate = [0.6348684210526315, 0.6736842105263158, 0.7125, 0.7519736842105263, 0.7901315789473684,0.8447368421052631, 0.8842105263157894, 0.9223684210526316, 0.9618421052631579]

    recans_obj = RecAnsArea(img_url,paperno)
    ans_path = recans_obj.get_ans_path()
    print "ans_path:",ans_path

    pap = ssPaper.objects.filter(paperno=paperno).first()
    pps = PaperPieces.objects.filter(paperno=paperno)\
            .filter(Q(quetype=0)|Q(quetype=1))

    std_point = get_std_point(img_url)
    print std_point,999999999999999999999
    #raise Exception

    if not std_point:
        std_point = std_point = eval(pap.std_point)[0] 

    ans_th_rate = eval(pap.std_ans_point_x)
    ans_xx_rate = eval(pap.std_ans_point_y)
    std_quenos = list(pps.values('queno','squeno'))
    std_quenos = [{'queno':int(x.get('queno')),'squeno':int(x.get('squeno'))} \
                for x in std_quenos]
    std_quenos = sort_quenos(std_quenos)
    std_quenos = [(x.get('queno','0'),x.get('squeno','0'))\
                    for x in std_quenos]
    #std_quenos = [(1, 0), (2, 0), (3, 0), (4, 0), (5, 0), (6, 0), (7, 0), (8, 0), (9, 0),(1, 0), (2, 0), (3, 0), (4, 0), (5, 0), (6, 0), (7, 0), (8, 0), (9, 0),(1, 0), (2, 0), (3, 0), (4, 0), (5, 0), (6, 0), (7, 0), (8, 0), (9, 0),(1, 0), (2, 0), (3, 0), (4, 0), (5, 0), (6, 0), (7, 0), (8, 0), (9, 0)]

    detectpaper = DetectPaper(ans_path,std_point,std_quenos,ans_th_rate,ans_xx_rate)
    print detectpaper.cal_kaohao_str()
    print detectpaper.cal_ans_lst()

    ###################################
    ##英语上下结构
    #detectpaper = DetectEnPaper(ans_path,std_point,std_quenos,ans_th_rate,ans_xx_rate)
    #print detectpaper.cal_kaohao_str()
    #print detectpaper.cal_ans_lst()
    ##print detectpaper.get_std_xy()



if __name__ == "__main__":
    main()
