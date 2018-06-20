#-*-coding:utf-8 -*-
import cv2
import numpy as np
from PIL import Image
import imutils
from imutils import contours
from imutils.perspective import four_point_transform
import os
import json


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

def get_median(data):
    data.sort()
    half = len(data) // 2
    return (data[half] + data[~half]) / 2

def dnoise_kh(lst=[]):
    '''
    '''
    new_lst = []
    for _l in lst:
        x = _l.get('x')
        if x < 50:
            continue
        new_lst.append(_l)
    return new_lst


def get_mid_repeat(lst=[]):
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
        


def denoise_data_bys(lst=[]):
    """算法面积去噪声
    """
    tmp = []
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


def denoise_data_byh(lst=[]):
    '''
    算法去噪声
    lst = [{'h':29,'w':110},{'h':26,'w':110},{'h':24,'w':110},
    {'h':2,'w':110},{'h':2,'w':110},{'h':3,'w':110}]
    '''
    lst = sorted(lst,key=lambda x:x['h'],reverse=True)
    mylog(lst=lst)

    h_lst = [x['h'] for x in lst]



    #计算相邻两个元素的差值

    dis_lst = []
    for i,v in enumerate(h_lst):
        if i < len(h_lst)-1:
            dis = h_lst[i]-h_lst[i+1]
            dis_lst.append(dis)

    import copy
    org_dis_lst = copy.deepcopy(dis_lst)
    
    #相邻差值倒序排列
    dis_lst.sort()
    dis_lst.reverse()
    dis_max = dis_lst[0] if dis_lst else 0
    #取中值
    mid_h = get_median(h_lst) if h_lst else 0

    mylog(dis_lst=dis_lst)
    mylog(mid_h=mid_h)
    if dis_max > mid_h/3 or True:
        #取出最大间距的索引
        index = org_dis_lst.index(dis_max)
        lst = lst[:index+1]

    return lst


def denoise_data_byw(lst):
    '''
    算法去噪声
    lst = [{'h':29,'w':110},{'h':26,'w':110},{'h':24,'w':110},
    {'h':2,'w':110},{'h':2,'w':110},{'h':3,'w':110}]
    '''
    lst = sorted(lst,key=lambda x:x['w'],reverse=True)
    h_lst = [x['w'] for x in lst]

    ##计算相邻两个元素的差值
    dis_lst = []
    for i,v in enumerate(h_lst):
        if i < len(h_lst)-1:
            dis = h_lst[i]-h_lst[i+1]
            dis_lst.append(dis)
    import copy
    org_dis_lst = copy.deepcopy(dis_lst)

    dis_lst.sort()
    dis_lst.reverse()

    #取中值
    mid_h = get_median(h_lst) if h_lst else 0
    dis_max = dis_lst[0] if dis_lst else 0

    mylog(dis_lst=dis_lst)
    mylog(mid_h=mid_h)

    if dis_max > mid_h/3 or True:
        index = org_dis_lst.index(dis_max)
        lst = lst[:index+1]

    return lst



class Paper:
    def __init__(self,image_path="qingxie.png",quenos=None,std_ans_y=None):
        '''
        具体试卷处理
        从试卷中找出答案区域
        '''

        self.quenos = quenos
        self.std_ans_y = std_ans_y
        #图片路径
        self.image_path = image_path
        #腐蚀后的图像
        self.erode_image = None
        #膨胀后的图像
        self.dilate_image = None

        #腐蚀与膨胀核大小
        #腐蚀核大小
        self.edsize = cv2.getStructuringElement(cv2.MORPH_CROSS,(5,10))
        #膨胀核大小
        self.dilsize = cv2.getStructuringElement(cv2.MORPH_CROSS,(6,6))
        
        #cv2图片对象
        self.image = cv2.imread(self.image_path)


        #图片的宽高度
        self.width = self.image.shape[1]
        self.height = self.image.shape[0]

        #图片转成灰度图
        self.gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        #做高斯模糊处理
        blurred = cv2.GaussianBlur(self.gray, (5, 5), 0)
        #边缘检测后的图
        self.edged = cv2.Canny(self.gray, 75, 200)

        #对灰度图进行二值化处理
        ret,thresh = cv2.threshold(self.gray,0,250,cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
        #ret,thresh = cv2.threshold(self.gray,0,250,cv2.THRESH_BINARY)
        #ret,thresh = cv2.threshold(self.gray,0,250,cv2.THRESH_TRUNC)
        #ret,thresh = cv2.threshold(self.gray,0,250,cv2.THRESH_TOZERO)
        #ret,thresh = cv2.threshold(self.gray,0,250,cv2.THRESH_TOZERO_INV)
        self.thresh = thresh

        ##输出边缘检测图像
        #spt_lst = os.path.splitext(image_path)
        #edged_path = spt_lst[0] + '_edged' + spt_lst[1]
        #cv2.imwrite(edged_path,self.edged)
        ##输出thresh二值图
        #thresh_path = spt_lst[0] + '_thresh' + spt_lst[1]
        #cv2.imwrite(thresh_path,self.thresh)

    def _erode(self):
        '''
        图像腐蚀操作
        '''
        self.erode_image = cv2.erode(self.thresh,self.edsize)
        ##输出腐蚀后的图像
        #spt_lst = os.path.splitext(self.image_path)
        #erode_path = spt_lst[0] + '_erode' + spt_lst[1]
        #cv2.imwrite(erode_path,self.erode_image)
        pass


    def _dilate(self):
        '''
        图像膨胀操作
        '''
        #self.dilate_image = cv2.erode(self.thresh,self.edsize)
        self.dilate_image = cv2.dilate(self.thresh,self.dilsize)
        #输出膨胀后的图像
        spt_lst = os.path.splitext(self.image_path)
        dilate_path = spt_lst[0] + '_dilate_cnts' + spt_lst[1]
        cv2.imwrite(dilate_path,self.dilate_image)

    def get_ans_point(self,std_point=None):
        '''
        从试卷中提取答案区域
        获取答案区域定位点
        '''
        #cnts = cv2.findContours(self.edged, cv2.RETR_CCOMP , cv2.CHAIN_APPROX_SIMPLE)
        #cnts = cv2.findContours(self.edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        #cnts = cv2.findContours(self.thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        #cnts = cv2.findContours(self.thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        #cnts = cv2.findContours(self.edged, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        self._dilate()
        #cnts = cv2.findContours(self.thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        #cnts = cv2.findContours(self.thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cv2.findContours(self.dilate_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        #cnts = cv2.findContours(self.dilate_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        #print cnts
        all_attr = cnts[1]
        cnts = cnts[0] if imutils.is_cv2() else cnts[1]

        #print cnts,33333333333333333

        if len(cnts) > 0:
            cnts = sorted(cnts, key=cv2.contourArea, reverse=True)

            all_c = []
            for c in cnts:
                peri = 0.01*cv2.arcLength(c, True)

                approx = cv2.approxPolyDP(c, peri, True)
                (x, y, w, h) = cv2.boundingRect(c)
                if w > self.width*0.25 and x>50 and y>100:
                    print _c,9999999999999999
                if len(approx)==4:
                    (x, y, w, h) = cv2.boundingRect(c)
                    ##标识出识别出的机型轮廓并输出
                    #cv2.drawContours(self.image,c,-1,(255,0,255),3)
                    #spt_lst = os.path.splitext(self.image_path)
                    #draw_path = spt_lst[0] + '_draw' + spt_lst[1]
                    #cv2.imwrite(draw_path,self.image)
                    #break
                    if h > self.height*0.125:
                        continue
                    if w > self.width*0.25 and x>50 and y>100:
                        #print 5555555555555555

                        #标识出识别出的机型轮廓并输出
                        cv2.drawContours(self.image,c,-1,(255,0,255),3)
                        spt_lst = os.path.splitext(self.image_path)
                        draw_path = spt_lst[0] + '_draw' + spt_lst[1]
                        cv2.imwrite(draw_path,self.image)

                        all_c.append({'x':x,'y':y,'w':w,'h':h})

            all_c = sorted(all_c,key=lambda x:x['w'],reverse=True)
            print all_c,1111111111111
            for _c in all_c:
                x = _c.get('x')
                y = _c.get('y')
                w = _c.get('w')
                h = _c.get('h')
                #将提取到的点进行识别
                cut_path = self.cut_ans_area((x,y,x+w,y+h))
                ansdetect = AnsDetect(cut_path)

                std_w = std_point.get('w')
                std_h = std_point.get('h')
                
                #腐蚀核大小
                edsize = cv2.getStructuringElement(cv2.MORPH_RECT,(std_w/4,std_h/6))

                dilsize = cv2.getStructuringElement(cv2.MORPH_RECT,(std_w/4,std_h/6))

                dilsize_join = cv2.getStructuringElement(cv2.MORPH_CROSS,(std_h/10,std_h/10))
                try:
                    x_rate,y_rate = ansdetect.recStdXY(edsize,dilsize,dilsize_join)
                except Exception,e:
                    import traceback
                    traceback.print_exc()
                    x_rate,y_rate = None,None
                    continue
                #if x_rate and len(y_rate)>3:
                if len(x_rate)>5:
                    return (x,y,x+w,y+h)
                    return (x+5,y+5,x+w-5,y+h-10)

        else:
            return None

    def _line(self):
        '''
        '''
        minLineLength = 20
        maxLineGap = 10
        lines = cv2.HoughLinesP(self.edged,np.pi/180,118,minLineLength,maxLineGap)
        for x1,y1,x2,y2 in lines[0]:
            cv2.line(self.image,(x1,y1),(x2,y2),(0,255,0),1)
            #输出答题卡区域图像
            ans_lst = os.path.splitext(self.image_path)
            ans_path = ans_lst[0] + '_ansarea' + ans_lst[1]
            cv2.imwrite(ans_path,self.image)

    def cut_ans_area(self,cut_point=None):
        '''
        从试卷裁剪出答题卡区域图片
        '''
        #print cut_point,66666666666666666666666
        img = Image.open(self.image_path)
        region = img.crop(cut_point)

        spt_lst = os.path.splitext(self.image_path)
        cut_path = spt_lst[0] + '_cut' + spt_lst[1]

        region.save(cut_path)

        return cut_path



class AnsDetect:
    '''
    答题卡区域选项识别
    '''

    def __init__(self,image_path="mark.png",quenos=None,std_ans_y=None):
        '''
        '''

        self.quenos = quenos
        self.std_ans_y = std_ans_y

        self.image_path = image_path
        self.erode_image = None
        self.dilate_image = None
        self.kh_std_x = None
        self.kh_std_y = None
        self.kh_std_w = None
        self.kh_std_h = None
        #腐蚀与膨胀核大小
        #self.edsize = cv2.getStructuringElement(cv2.MORPH_CROSS,(8,8))
        #self.edsize = cv2.getStructuringElement(cv2.MORPH_RECT,(6,10))
        #腐蚀
        #self.edsize = cv2.getStructuringElement(cv2.MORPH_RECT,(10,8))
        self.edsize = cv2.getStructuringElement(cv2.MORPH_RECT,(25,10))
        #膨胀为提取框
        self.cnts_dilsize = cv2.getStructuringElement(cv2.MORPH_CROSS,(5,5))
        self.thresh_dilsize = cv2.getStructuringElement(cv2.MORPH_RECT,(2,2))
        #膨胀为提取答案
        self.ans_dilsize = cv2.getStructuringElement(cv2.MORPH_RECT,(25,30))

        self.image = cv2.imread(self.image_path)

        #图片的宽度
        self.width = self.image.shape[1]
        self.height = self.image.shape[0]

        self.gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(self.gray, (5, 5), 0)
        #edged = cv2.Canny(blurred, 75, 200)
        self.edged = cv2.Canny(self.gray, 75, 200)
        #对灰度图进行二值化处理
        #self.thresh = cv2.threshold(self.gray,200,255,cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
        self.thresh = cv2.threshold(self.gray.copy(),200,255,cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

        ##输出边缘检测图像
        spt_lst = os.path.splitext(image_path)
        edged_path = spt_lst[0] + '_edged' + spt_lst[1]
        cv2.imwrite(edged_path,self.edged)

        #输出thresh二值图
        thresh_path = spt_lst[0] + '_thresh' + spt_lst[1]
        cv2.imwrite(thresh_path,self.thresh)

    def img_to_thresh(self,img=None):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edged = cv2.Canny(gray, 75, 200)
        #对灰度图进行二值化处理
        thresh = cv2.threshold(gray.copy(),200,255,cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

        return thresh

    def erode(self,img=None,edsize=None):
        erode_img = cv2.erode(img,edsize)
        #输出腐蚀后的图像
        spt_lst = os.path.splitext(self.image_path)
        erode_path = spt_lst[0] + '_erode_new' + spt_lst[1]
        cv2.imwrite(erode_path,erode_img)

        return erode_img


    def dilate(self,img=None,dilsize=None):
        dilate_img = cv2.dilate(img,dilsize)
        #输出腐蚀后的图像
        spt_lst = os.path.splitext(self.image_path)
        dil_path = spt_lst[0] + '_dilate_new' + spt_lst[1]
        cv2.imwrite(dil_path,dilate_img)

        return dilate_img


    def _erode(self,edsize=None):
        '''
        图像腐蚀操作
        '''
        self.erode_image = cv2.erode(self.thresh,edsize)
        #输出腐蚀后的图像
        spt_lst = os.path.splitext(self.image_path)
        erode_path = spt_lst[0] + '_erode' + spt_lst[1]
        cv2.imwrite(erode_path,self.erode_image)
        pass

    def _dilate_thresh(self):
        self.thresh = cv2.dilate(self.thresh,self.thresh_dilsize)
        #输出膨胀后的图像
        spt_lst = os.path.splitext(self.image_path)
        dilate_path = spt_lst[0] + '_dilate_thresh' + spt_lst[1]
        cv2.imwrite(dilate_path,self.thresh)


    def _dilate(self,cnts_dilsize=None):
        '''
        图像膨胀操作
        '''
        self.cnts_dilate_image = cv2.dilate(self.thresh,self.cnts_dilsize)
        #self.dilate_image = cv2.dilate(self.erode_image,self.edsize1)
        #输出膨胀后的图像
        spt_lst = os.path.splitext(self.image_path)
        dilate_path = spt_lst[0] + '_dilate_cnts' + spt_lst[1]
        cv2.imwrite(dilate_path,self.cnts_dilate_image)

    def _erode_dilate(self,dilsize=None):
        '''
        图像膨胀操作
        '''
        #self.dilate_image = cv2.erode(self.thresh,self.edsize)
        self.dilate_image = cv2.dilate(self.erode_image,dilsize)
        #输出膨胀后的图像
        spt_lst = os.path.splitext(self.image_path)
        dilate_path = spt_lst[0] + '_dilate' + spt_lst[1]
        cv2.imwrite(dilate_path,self.dilate_image)


    def findKaoHaoCnts(self,std_point,ans_th_rate):
        '''
        '''

        std_w = std_point.get('w')

        kh_max_x =  self.width*ans_th_rate[0]-std_w

        kh_std_dct = {'x':kh_max_x,'y':0,'w':0,'h':0}

        print kh_std_dct

        #return  None

        kh_std_lst = []

        #膨胀凸显外侧轮廓
        self._dilate()

        #img,cnts,hierarchy = cv2.findContours(self.cnts_dilate_image.copy(),cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        img,cnts,hierarchy = cv2.findContours(self.thresh.copy(),cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        #img,cnts,hierarchy = cv2.findContours(self.edged.copy(),cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        if len(cnts) > 0:
            cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
            
            #mylog(cnts=cnts)
            for i,c in enumerate(cnts):
                #轮廓逼近
                peri = cv2.arcLength(c, True)
                approx = cv2.approxPolyDP(c, 0.01 * peri, True)
                (x, y, w, h) = cv2.boundingRect(c)

                if h > self.height/3:
                    continue

                if w < self.width/8:
                    continue

                if x > self.width/3:
                    continue

                kh_std_lst.append({'x':x,'y':y,'w':w,'h':h})



            kh_std_lst = sorted(kh_std_lst,key=lambda x:x['y'])

            mylog(kh_std_lst=kh_std_lst)
            if kh_std_lst:
                kh_std_dct = kh_std_lst[len(kh_std_lst)/2]
            else:
                kh_std_dct = None
                kh_std_dct = {'x':kh_max_x-self.height*2,'y':0,'w':self.height*2,'h':0}

            return kh_std_dct

        else:
            return None

    
    def findFillCnts(self,kh_std_dct=None,edsize=None,dilsize=None,
                dilsize_join=None,std_point=None):
        '''
        提取已经填图好的考号和答案轮廓,并进行区分
        '''
        std_point_w = std_point.get("w")
        #考号基准分割线
        kh_std_x = kh_std_dct.get('x')
        kh_std_w = kh_std_dct.get('w')

        kh_max_x = kh_std_x + kh_std_w 


        djimg = self.dilate(self.thresh,dilsize_join)
        #腐蚀操作
        eimg = self.erode(djimg,edsize)
        #膨胀操作
        dilimg = self.dilate(eimg,dilsize)

        #提取填图好的轮廓
        img,cnts,hierarchy = cv2.findContours(dilimg,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

        fill_cnts_lst = []
        kh_cnts_lst = []
        ans_cnts_lst = []

        if len(cnts) > 0:
            cnts = sorted(cnts, key=cv2.contourArea, reverse=True)

            for i,c in enumerate(cnts):
                #轮廓逼近
                peri = cv2.arcLength(c, True)
                approx = cv2.approxPolyDP(c, 0.01 * peri, True)
                (x, y, w, h) = cv2.boundingRect(c)

                fill_cnts_lst.append({'x':x,'y':y,'w':w,'h':h})

            mylog(fill_cnts_lst=fill_cnts_lst)
            print 444444444444444444444444444444444444444444444444
            #raise Exception
            #将提取到的轮廓按照x排序
            fill_cnts_lst = sorted(fill_cnts_lst,key=lambda x:x['x'])
            for _cdct in fill_cnts_lst:
                _x = _cdct.get('x')
                _y = _cdct.get('y')
                _w = _cdct.get('w')

                #if _w > std_point_w*3:
                #    pass

                if not _x:
                    continue
                #if not _y:
                #    continue

                if _cdct.get('x') < kh_max_x:
                    #print kh_std_x,666666666666666666
                    ##在考号外面的为干扰因素
                    if _x < kh_std_x:
                        print 666666666666666666666666666
                        print _cdct
                        continue
                    else:
                        kh_cnts_lst.append(_cdct)
                else:
                    ans_cnts_lst.append(_cdct)

            kh_cnts_lst = sorted(kh_cnts_lst,key=lambda x:x["w"],reverse=True)
            ans_cnts_lst = sorted(ans_cnts_lst,key=lambda x:x["w"],reverse=True)
            print u"考号清洗前:"
            mylog(kh_cnts_lst=kh_cnts_lst)
            if len(kh_cnts_lst)>6:
                kh_cnts_lst = denoise_data_bys(kh_cnts_lst)
            ans_cnts_lst = denoise_data_bys(ans_cnts_lst)
            print u"考号清洗后:"
            mylog(kh_cnts_lst=kh_cnts_lst)
            mylog(ans_cnts_lst=ans_cnts_lst)
            #print 5555555555555555555555555555555555555555555555555555
            #raise Exception

            return kh_cnts_lst,ans_cnts_lst

        #raise Exception

        return kh_cnts_lst,ans_cnts_lst







    def recKaoHao(self,kh_std_dct=None,edsize=None,dilsize=None,
                    dilsize_join=None,std_point=None):
        '''
        识别考号
        kh_cnts:填图好的考号轮廓,考号区域标准轮廓信息
        '''
        #考号基准分割线
        kh_std_x = kh_std_dct.get('x')
        kh_std_w = kh_std_dct.get('w')

        kh_max_x = kh_std_x + kh_std_w 

        kh_max_x = kh_std_x

        print std_point,33333333333333333333
        #raise Exception

        #获取考号和答案轮廓信息
        kh_cnts_lst,_ = self.findFillCnts(kh_std_dct,edsize,dilsize,
                        dilsize_join,std_point)

        mylog(kh_cnts_lst=kh_cnts_lst)
        #print 6666666666666666666
        #raise Exception
        #if len(kh_cnts_lst) > 6:
        #    kh_cnts_lst = denoise_data_bys(kh_cnts_lst)

        #卡号填图区域的平均宽度
        avg_w = sum([x['w'] for x in kh_cnts_lst])/len(kh_cnts_lst) \
                if kh_cnts_lst else 0

        #计算基准大小
        meta_w = (kh_std_w - avg_w*10)/11

        #考号轮廓按Y排序
        kh_cnts_lst = sorted(kh_cnts_lst,key=lambda x:x['y'])

        mylog(kh_cnts_lst=kh_cnts_lst)
        #print 99999999999999999999999999999
        #raise Exception

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


    def recAns(self,kh_std_dct=None,ans_th_rate=[],ans_xx_rate=[],\
            std_quenos=[],edsize=None,dilsize=None,dilsize_join=None,std_point=None):
        '''
        识别选择题答案
        ans_th_rate:选项基准比例,ans_xx_rate:题号基准比例
        std_quenums:总提数
        '''
        std_quenos.sort()

        char_lst = ["A","B","C","D","E","F","G","H","I"]
        #计算对比比例
        #th_rate_dis = (ans_th_rate[-1]-ans_th_rate[0])/len(ans_th_rate)
        th_rate_dis = (ans_th_rate[-1]-ans_th_rate[0])/len(std_quenos)
        th_rate_dis = round(th_rate_dis,5)

        xx_rate_dis = (ans_xx_rate[-1]-ans_xx_rate[0])/len(char_lst)
        xx_rate_dis = round(xx_rate_dis,5)


        #设置标准题号坑
        std_ans_queno = []
        std_quenos.sort()
        mylog(std_quenos=std_quenos)
        for i,qn in enumerate(std_quenos):
            dct = {}
            dct['queno'] = str(std_quenos[i])
            dct['rate'] = ans_th_rate[i]
            dct['cnts'] = []
            dct['range'] = self.width*ans_th_rate[i]
            std_ans_queno.append(dct)

        mylog(std_ans_queno=std_ans_queno)


        #是指标准选项标靶
        std_ans_xx = []
        for i,xx in enumerate(ans_xx_rate):
            dct = {}
            dct['rate'] = ans_xx_rate[i] 
            dct["xx"] = char_lst[i] 
            std_ans_xx.append(dct)


        #获取考号和答案轮廓信息
        _,ans_cnts_lst = self.findFillCnts(kh_std_dct,edsize,dilsize,
                    dilsize_join,std_point)

        mylog(ans_cnts_lst=ans_cnts_lst)
        #raise Exception


        #去除干扰因素
        ans_cnts_lst = filter(lambda x:x['h']<self.height/8,ans_cnts_lst)

        ans_cnts_lst = denoise_data_bys(ans_cnts_lst)

        #ans_cnts_lst.append({'h':0,'w':0,'x':0,'y':0})
        #ns_cnts_lst =denoise_data_byh(ans_cnts_lst)
        #if len(ans_cnts_lst)>len(std_quenos):
        #    ans_cnts_lst.append({'h':0,'w':0,'x':0,'y':0})
        #    #按w去除噪声
        #    ans_cnts_lst =denoise_data_byw(ans_cnts_lst)
        mylog(ans_cnts_lst=ans_cnts_lst)

        #把答案轮廓按x排序
        ans_cnts_lst = sorted(ans_cnts_lst,key=lambda x:x['x'])

        mylog(ans_cnts_lst=ans_cnts_lst)
        
        #取出轮廓里面的最小宽度
        min_w = [x['w'] for x in ans_cnts_lst][0]
        avg_w = sum([x['w'] for x in ans_cnts_lst])/len(ans_cnts_lst) if len(ans_cnts_lst) else 0

        #相邻X间距小于1/2最小宽度最归结为一组
        import collections
        julei = []
        dct = collections.OrderedDict()
        #ans_cnts_lst.append({'h':0,'w':0,'x':0,'y':0})
        for _a in ans_cnts_lst:
            w = _a.get('w')
            x = _a.get('x')
            k = dct.keys()[-1] if dct.keys() else 0

            #if x-k < min_w:
            if x-k < w/2:
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


            ###############################################
            ##测试代码
            #x = julei[i].get('x')
            #w = julei[i].get('w')

            #w = get_mid_repeat(julei)

            #if len(std_ans_queno) > 20:
            #    x = x
            #else:
            #    x = x+w/3.5
            ##_std_th_range = [self.width*th_rate+10-w/1.5,self.width*th_rate+10+w/1.1]
            ##_std_th_range = [self.width*th_rate-avg_w/1.2,self.width*th_rate+avg_w/1.1]

            #if len(std_ans_queno) > 20:
            #    _std_th_range = [self.width*th_rate-w/2,self.width*th_rate+w/2]
            #else:
            #    _std_th_range = [self.width*th_rate-w/2,self.width*th_rate+w/2]

            #print u'x范围:'
            #print x
            #print _std_th_range
            #print '\n\n'
            #continue
            ###############################################


            try:

                for j,_jl in enumerate(julei):
                    _cnts = julei[j].get('cnts')

                    x = julei[j].get('x')

                    w = julei[j].get('w')

                    w = get_mid_repeat(julei)

                    if len(julei) >= 20:
                        x = x+w/2
                    else:
                        x = x+w/3.5

                    ##第I提x的标准范围
                    #_std_th_range = [self.width*th_rate-w/1.5,self.width*th_rate+w/1.5]
                    #_std_th_range = [self.width*th_rate-avg_w/1.2,self.width*th_rate+avg_w/1.1]

                    if len(julei) >= 20:
                        _std_th_range = [self.width*th_rate-w/2,self.width*th_rate+w/2]
                    else:
                        _std_th_range = [self.width*th_rate-w/2,self.width*th_rate+w/2]

                    #if j==2:
                    #    print x
                    #    print std_que
                    #    print _std_th_range,99999999999
                    #    break

                    if x>=_std_th_range[0] and x<=_std_th_range[1]:
                        std_ans_queno[i]['cnts'].extend(_cnts)
                        julei.remove(_jl)
                    #else:
                    #    print u'x范围:'
                    #    print std_ans_queno[i]['queno']
                    #    print x
                    #    print _std_th_range
                    #    print '\n\n'
            except Exception ,e:
                import traceback
                traceback.print_exc()
                pass

        mylog(std_ans_queno=std_ans_queno)


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
                dis_y =  ans_xx_rate[0]*self.height


                #选项高度
                xx_dis = self.height-dis_y
                #计算平均高度
                avg_h = sum(x['h'] for x in _cnts)/len(_cnts)
                #计算间隙的高度
                meta_h = (xx_dis - avg_h*len(ans_xx_rate))/(len(ans_xx_rate)-1)

                #print u'题号:'
                #print que_cnt_dct['queno']

                for c in _cnts:
                    ##通过距离在范围内计算
                    y = c.get('y')
                    h = c.get('h')

                    y = y+avg_h/4

                    std_y_ranges = [[self.height*x-avg_h/1.2,self.height*x+avg_h/1.2] for x in ans_xx_rate]
                    std_y_ranges = [[self.height*x-avg_h/1.1,self.height*x+avg_h/1.1] for x in ans_xx_rate]
                    
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

        return shit_lst


    def recStdXY(self,edsize=None,dilsize=None,dilsize_join=None):
        '''
        英语标准填图识别
        '''
        #读取答案轮廓
        #djimg = self.dilate(self.thresh,dilsize_join)
        #腐蚀操作
        #eimg = self.erode(djimg,edsize)
        eimg = self.erode(self.thresh,edsize)
        #膨胀操作
        dilimg = self.dilate(eimg,dilsize)

        img,cnts,hierarchy = cv2.findContours(dilimg,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

        ans_cnts_lst = []

        if len(cnts) > 0:
            cnts = sorted(cnts, key=cv2.contourArea, reverse=True)

            for i,c in enumerate(cnts):
                #轮廓逼近
                peri = cv2.arcLength(c, True)
                approx = cv2.approxPolyDP(c, 0.01 * peri, True)
                (x, y, w, h) = cv2.boundingRect(c)

                ans_cnts_lst.append({'x':x,'y':y,'w':w,'h':h})

        ans_cnts_lst.append({'x':0,'y':0,'w':0,'h':0})
        ans_cnts_lst = denoise_data_byh(ans_cnts_lst)
        ans_cnts_lst.append({'x':0,'y':0,'w':0,'h':0})
        ans_cnts_lst = denoise_data_byw(ans_cnts_lst)

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
            #y_rate.append(th_y/float(self.height))
            y_rate.append((th_y+th_h/2)/float(self.height))

        return x_rate,y_rate




    def _line(self):
        '''
        直线检测
        '''
        minLineLength = 200
        maxLineGap = 100
        lines = cv2.HoughLinesP(self.thresh,1,np.pi/180,118,minLineLength,maxLineGap)
        for line in lines[:,:,:]:
            x1,y1,x2,y2 = line[0,:]
            #print x1,y1,x2,y2
            cv2.line(self.image,(x1,y1),(x2,y2),(0,255,0),1)
            #输出答题卡区域图像
            ans_lst = os.path.splitext(self.image_path)
            ans_path = ans_lst[0] + '_line' + ans_lst[1]
            cv2.imwrite(ans_path,self.image)



if __name__ == "__main__":
    #img_path = '010114.jpg'
    #img_path = '010102.jpg'
    #img_path = '010109.jpg'
    #img_path = '010110.jpg'
    #img_path = '010114.jpg'
    #img_path = '010118.jpg'
    img_path = '010117.jpg'
    #img_path = '010103.jpg'
    img_path = os.path.abspath(img_path)
    paper = Paper(img_path)
    #答案答题卡区域
    ans_area_point = paper.get_ans_point()
    if ans_area_point:
        ans_img_path = paper.cut_ans_area(ans_area_point)
        
        #检测和分析答题卡
        ansdetect = AnsDetect(ans_img_path)
        #腐蚀操作
        ansdetect._erode()
        #膨胀操作
        ansdetect._erode_dilate()
        ##直线检测
        #ansdetect._line()

        #考号区域标准信息提取
        kh_std_dct = ansdetect.findKaoHaoCnts()
        kh_max_x = kh_std_dct.get('x') + kh_std_dct.get('w')

        #提取填图的轮廓
        kh_cnts_lst,ans_cnts_lst = ansdetect.findFillCnts(kh_max_x)

        #识别考号
        kh_str = ansdetect.recKaoHao(kh_std_dct)
        #print kh_str

        #识别选择题答案
        #print ansdetect.recAns(kh_std_dct)


