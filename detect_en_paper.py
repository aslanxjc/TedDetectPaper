#-*-coding:utf-8 -*-
#from new_detect_image import Paper,AnsDetect
import cv2
import os
import collections

os.environ.setdefault("DJANGO_SETTINGS_MODULE","ss_mark.settings")

from ss_mark.models import Paper as ssPaper,PaperPieces
from django.db.models import Q
from utils.DetectPaper.detect_en_image import Paper,EnAnsDetect

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


def get_ans_area(img_path=None):
    #img_path = '010121.jpg'
    img_path = os.path.abspath(img_path)
    paper = Paper(img_path)
    #答案答题卡区域
    #kh_area_point,ans_area_point = paper.get_std_ans_point()
    kh_area_point,ans_area_point = paper.get_ans_point()

    print kh_area_point,111111111111111111
    print ans_area_point,222222222222222222

    if ans_area_point:
        kh_img_path = paper.cut_kh_area(kh_area_point)
        ans_img_path = paper.cut_ans_area(ans_area_point)

    return kh_img_path,ans_img_path

def get_std_ans_area(img_path=None):
    #img_path = '010121.jpg'
    img_path = os.path.abspath(img_path)
    paper = Paper(img_path)
    #答案答题卡区域
    kh_area_point,ans_area_point = paper.get_std_ans_point()
    #kh_area_point,ans_area_point = paper.get_ans_point()

    if ans_area_point:
        #kh_img_path = paper.cut_kh_area(kh_area_point)
        ans_img_path = paper.cut_ans_area(ans_area_point)

    return ans_img_path


def rec_kaohao_str(kh_img_path=None,std_point=None):
    '''
    从答题卡识别出考号
    '''
    ansdetect = EnAnsDetect(kh_img_path)

    #考号区域标准信息提取
    #kh_std_dct = ansdetect.findKaoHaoCnts()

    #腐蚀核计算方法std_w/2,std_h/3,链接核计算方法std_w/6,std_h/6
    std_w = std_point.get('w')
    std_h = std_point.get('h')
    
    #腐蚀核大小
    #edsize = cv2.getStructuringElement(cv2.MORPH_RECT,(std_w/3,std_h/3))
    edsize = cv2.getStructuringElement(cv2.MORPH_RECT,(std_w/4,std_h/4))

    #dilsize = cv2.getStructuringElement(cv2.MORPH_RECT,(std_w/2,std_h/2))
    #dilsize = cv2.getStructuringElement(cv2.MORPH_RECT,(std_w/5,std_h/5))
    dilsize = cv2.getStructuringElement(cv2.MORPH_RECT,(std_w/8,std_h/8))

    dilsize_join = cv2.getStructuringElement(cv2.MORPH_CROSS,(std_w/8,std_w/8))
    dilsize_join = cv2.getStructuringElement(cv2.MORPH_CROSS,(std_w/10,std_w/10))

    ##检测和分析答题卡
    enansdetect = EnAnsDetect(kh_img_path)
    #识别考号
    kh_str = enansdetect.recKaoHao(edsize,dilsize,dilsize_join)

    return kh_str


def rec_shit_ans(ans_img_path=None,ans_th_rate=[],ans_xx_rate=[],std_quenos=[],std_point=None):
    '''
    从答题卡识别出选择题答案
    ans_th_rate:题号标准比率
    ans_xx_rate:选项标准比率
    std_quenos:选择题题号
    '''
    ansdetect = EnAnsDetect(ans_img_path)

    #考号区域标准信息提取

    #腐蚀核计算方法std_w/2,std_h/3,链接核计算方法std_w/6,std_h/6
    std_w = std_point.get('w')
    std_h = std_point.get('h')
    
    #腐蚀核大小
    edsize = cv2.getStructuringElement(cv2.MORPH_RECT,(std_w/2,std_h/3))
    edsize = cv2.getStructuringElement(cv2.MORPH_RECT,(std_w/3,std_h/4))
    #edsize = cv2.getStructuringElement(cv2.MORPH_RECT,(std_w/4,std_h/5))

    dilsize = cv2.getStructuringElement(cv2.MORPH_RECT,(std_w/2,std_h/3))
    #dilsize = cv2.getStructuringElement(cv2.MORPH_RECT,(std_w/3,std_h/4))
    dilsize = cv2.getStructuringElement(cv2.MORPH_RECT,(std_w/4,std_h/5))

    #dilsize_join = cv2.getStructuringElement(cv2.MORPH_CROSS,(std_h/8,std_h/8))
    dilsize_join = cv2.getStructuringElement(cv2.MORPH_CROSS,(std_h/10,std_h/10))

    #识别选择题答案
    shit_ans_lst = ansdetect.recAns(ans_th_rate,ans_xx_rate,std_quenos,edsize,dilsize,dilsize_join)

    return shit_ans_lst

def get_std_xy(ans_img_path=None,std_point=None):
    '''
    识别标准填图
    '''
    ansdetect = EnAnsDetect(ans_img_path)

    #考号区域标准信息提取

    #腐蚀核计算方法std_w/2,std_h/3,链接核计算方法std_w/6,std_h/6
    std_w = std_point.get('w')
    std_h = std_point.get('h')
    
    #腐蚀核大小
    edsize = cv2.getStructuringElement(cv2.MORPH_RECT,(std_w/2,std_h/3))

    dilsize = cv2.getStructuringElement(cv2.MORPH_RECT,(std_w/2,std_h/3))

    dilsize_join = cv2.getStructuringElement(cv2.MORPH_CROSS,(std_h/8,std_h/8))

    x_rate,y_rate = ansdetect.recStdXY(edsize,dilsize,dilsize_join)

    x_rate.sort()
    y_rate.sort()

    return x_rate,y_rate


def sort_quenos(std_quenos):
    '''
    '''
    new_std_quenos = []

    if std_quenos:
        std_quenos = sorted(std_quenos,key=lambda x:x['queno'])

        tmp = collections.OrderedDict()
        for i,_sq in enumerate(std_quenos):
            queno = _sq.get('queno','0')
            if not tmp.has_key(queno):
                tmp[queno] = [_sq]
            else:
                tmp[queno].append(_sq)

        for k,v in tmp.items():
            v = sorted(v,key=lambda x:x['squeno'])
            new_std_quenos.extend(v)

    return new_std_quenos
            


#def get_en_kh_ans(img_path,paperno,std_point):
def get_en_kh_ans(img_path,paperno):
    '''
    '''
    kh_img_path,ans_img_path = get_ans_area(img_path)

    pap = ssPaper.objects.filter(paperno=paperno).first()
    pps = PaperPieces.objects.filter(paperno=paperno).filter(Q(quetype=0)|Q(quetype=1))
    if pps:
        #std_quenos = list(pps.values_list('queno',flat=True))
        #std_quenos = [int(x) for x in std_quenos]
        #std_quenos.sort()
        std_quenos = list(pps.values('queno','squeno'))
        std_quenos = [{'queno':x.get('queno').zfill(3),'squeno':int(x.get('squeno'))} for x in std_quenos]
        std_quenos = sort_quenos(std_quenos)
        std_quenos = [(x.get('queno','0'),x.get('squeno','0')) for x in std_quenos]
    if pap:
        ans_th_rate = eval(pap.std_ans_point_x)
        ans_xx_rate = eval(pap.std_ans_point_y)

        std_point = eval(pap.std_point)[0]
        #std_point = std_point 

        kh_str = rec_kaohao_str(kh_img_path,std_point)
        mylog(kh_str=kh_str)

        shit_ans_lst = rec_shit_ans(ans_img_path,ans_th_rate,ans_xx_rate,std_quenos,std_point)
        mylog(shit_ans_lst=shit_ans_lst)
	
	return kh_str,shit_ans_lst



if __name__ == "__main__": 
    img_path = "/mnt/ss_mark_deploy/ss-mark/utils/DetectPaper/010001.jpg"
    paperno = "XZB2018RJYYB01001"
    get_en_kh_ans(img_path,paperno)

