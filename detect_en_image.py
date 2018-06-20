#-*-coding:utf-8 -*-
import cv2
import numpy as np
from PIL import Image
import imutils
from imutils import contours
from imutils.perspective import four_point_transform
import os
import json
from new_detect_image import denoise_data_byh,denoise_data_byw


def denoise_data_byw(lst):
    '''
    算法去噪声
    lst = [{'h':29,'w':110},{'h':26,'w':110},{'h':24,'w':110},
    {'h':2,'w':110},{'h':2,'w':110},{'h':3,'w':110}]
    '''
    lst = sorted(lst,key=lambda x:x['w'],reverse=True)
    #mylog(lst=lst)
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

    dis_max = dis_lst[0]
    index = org_dis_lst.index(dis_max)

    lst = lst[:index+1]

    return lst


def denoise_data_byh(lst=[]):
    '''
    算法去噪声
    lst = [{'h':29,'w':110},{'h':26,'w':110},{'h':24,'w':110},
    {'h':2,'w':110},{'h':2,'w':110},{'h':3,'w':110}]
    '''
    lst = sorted(lst,key=lambda x:x['h'],reverse=True)
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
    dis_max = dis_lst[0]
    #取出最大间距的索引

    index = org_dis_lst.index(dis_max)

    lst = lst[:index+1]

    return lst

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
        self.edsize = cv2.getStructuringElement(cv2.MORPH_CROSS,(5,20))
        #膨胀核大小
        self.dilsize = cv2.getStructuringElement(cv2.MORPH_CROSS,(2,2))
        
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
        self.thresh = thresh

        #输出边缘检测图像
        spt_lst = os.path.splitext(image_path)
        edged_path = spt_lst[0] + '_edged' + spt_lst[1]
        cv2.imwrite(edged_path,self.edged)
        #输出thresh二值图
        thresh_path = spt_lst[0] + '_thresh' + spt_lst[1]
        cv2.imwrite(thresh_path,self.thresh)

    def _erode(self):
        '''
        图像腐蚀操作
        '''
        self.erode_image = cv2.erode(self.thresh,self.edsize)
        #输出腐蚀后的图像
        spt_lst = os.path.splitext(self.image_path)
        erode_path = spt_lst[0] + '_erode' + spt_lst[1]
        cv2.imwrite(erode_path,self.erode_image)
        pass


    def _dilate(self):
        '''
        图像膨胀操作
        '''
        #self.dilate_image = cv2.erode(self.thresh,self.edsize)
        self.dilate_image = cv2.dilate(self.thresh,self.dilsize)
        #输出膨胀后的图像
        spt_lst = os.path.splitext(self.image_path)
        dilate_path = spt_lst[0] + '_dilate' + spt_lst[1]
        cv2.imwrite(dilate_path,self.dilate_image)

    def erode(self,img=None,edsize=None):
        erode_img = cv2.erode(img,edsize)
        #输出腐蚀后的图像
        spt_lst = os.path.splitext(self.image_path)
        erode_path = spt_lst[0] + '_erode' + spt_lst[1]
        cv2.imwrite(erode_path,erode_img)

        return erode_img


    def dilate(self,img=None,dilsize=None):
        dilate_img = cv2.dilate(img,dilsize)
        #输出腐蚀后的图像
        spt_lst = os.path.splitext(self.image_path)
        dil_path = spt_lst[0] + '_dilate' + spt_lst[1]
        cv2.imwrite(dil_path,dilate_img)

        return dilate_img

    def get_ans_point(self):
        '''
        从试卷中提取答案区域
        获取答案区域定位点
        '''
        self._dilate()

        cnts = cv2.findContours(self.dilate_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        all_attr = cnts[1]
        cnts = cnts[0] if imutils.is_cv2() else cnts[1]

        if len(cnts) > 0:
            cnts = sorted(cnts, key=cv2.contourArea, reverse=True)

            all_c = []
            for i,c in enumerate(cnts):
                peri = 0.01*cv2.arcLength(c, True)

                approx = cv2.approxPolyDP(c, peri, True)
                if len(approx)==4:
                    (x, y, w, h) = cv2.boundingRect(c)

                    ###标识出识别出的机型轮廓并输出
                    #cv2.drawContours(self.image,c,-1,(255,0,255),3)
                    #spt_lst = os.path.splitext(self.image_path)
                    #draw_path = spt_lst[0] + '_draw' + spt_lst[1]
                    #cv2.imwrite(draw_path,self.image)
                    print w,h,x,y,111111111111111111111
                    #print self.width/10

                    if i>10:
                        break
                    if w > self.width/10 and x>50 and y>100:
                        print 5555555555555555

                        #标识出识别出的机型轮廓并输出
                        cv2.drawContours(self.image,c,-1,(255,0,255),3)
                        spt_lst = os.path.splitext(self.image_path)
                        draw_path = spt_lst[0] + '_draw' + spt_lst[1]
                        cv2.imwrite(draw_path,self.image)

                        all_c.append({'x':x,'y':y,'w':w,'h':h})

            mylog(all_c=all_c)
            #return None
            print all_c,444444444444444444444

            all_c = sorted(all_c,key=lambda x:x['w'],reverse=True)

            #答案区域矩形切割点
            ans_area_point = all_c[0]
            ans_x = ans_area_point.get('x')
            ans_y = ans_area_point.get('y')
            ans_w = ans_area_point.get('w')
            ans_h = ans_area_point.get('h')
            ans_area_point = (ans_x,ans_y,ans_x+ans_w,ans_y+ans_h)
   
            #计算考号区域切割点高度小于100相邻差距不大于20的6个
            ##############################################
            kh_cnts = all_c[1:]

            kh_area = []
            #mylog(kh_cnts=kh_cnts)
            for _kc in kh_cnts:
                w = _kc.get('w')
                h = _kc.get('h')
                if h > 100:
                    continue
                if w/h < 10:
                    continue
                kh_area.append(_kc)
            ##############################################
            #mylog(kh_area=kh_area)

            kh_area = kh_area[:6]
            kh_area = sorted(kh_area,key=lambda x:x['y'])

            #mylog(kh_area=kh_area)
            #计算考号区域的切割点
            x0 = kh_area[0].get('x')
            y0 = kh_area[0].get('y') 
            #yo等于答案区域y-6*cnts[0]['h']
            y0 = all_c[0].get('y')-kh_area[0].get('h')*6
            #print x0,y0
            x1 = x0 + kh_area[0].get('w')
            y1 = y0 + kh_area[0].get('h')*6

            kh_area_point = (x0,y0,x1,y1)


            return kh_area_point,ans_area_point

        else:
            return None


    def get_std_ans_point(self):
        '''
        从试卷中提取答案区域
        获取答案区域定位点
        '''
        self._dilate()

        cnts = cv2.findContours(self.dilate_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        all_attr = cnts[1]
        cnts = cnts[0] if imutils.is_cv2() else cnts[1]

        if len(cnts) > 0:
            cnts = sorted(cnts, key=cv2.contourArea, reverse=True)

            all_c = []
            for i,c in enumerate(cnts):
                peri = 0.01*cv2.arcLength(c, True)

                approx = cv2.approxPolyDP(c, peri, True)
                if len(approx)==4:
                    (x, y, w, h) = cv2.boundingRect(c)

                    ###标识出识别出的机型轮廓并输出
                    #cv2.drawContours(self.image,c,-1,(255,0,255),3)
                    #spt_lst = os.path.splitext(self.image_path)
                    #draw_path = spt_lst[0] + '_draw' + spt_lst[1]
                    #cv2.imwrite(draw_path,self.image)
                    if i>10:
                        break
                    
                    #识别用
                    #if w > self.width/6.0 and x>50 and y>100:
                    #获取标准填图
                    if w > self.width/3.0 and x>50 and y>100 and h > 100:
                        all_c.append({'x':x,'y':y,'w':w,'h':h})

            all_c = sorted(all_c,key=lambda x:x['w'],reverse=True)

            ans_area_point = all_c[0]
            ans_x = ans_area_point.get('x')
            ans_y = ans_area_point.get('y')
            ans_w = ans_area_point.get('w')
            ans_h = ans_area_point.get('h')

            kh_area_point = ()

            ans_area_point = (ans_x,ans_y,ans_x+ans_w,ans_y+ans_h)

            return kh_area_point,ans_area_point

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

    def cut_kh_area(self,cut_point=None):
        '''
        从试卷裁剪出答题卡区域图片
        '''
        #print cut_point,66666666666666666666666
        img = Image.open(self.image_path)
        region = img.crop(cut_point)

        spt_lst = os.path.splitext(self.image_path)
        cut_path = spt_lst[0] + '_cutkh' + spt_lst[1]

        region.save(cut_path)

        return cut_path

    def cut_ans_area(self,cut_point=None):
        '''
        从试卷裁剪出答题卡区域图片
        '''
        #print cut_point,66666666666666666666666
        img = Image.open(self.image_path)
        region = img.crop(cut_point)

        spt_lst = os.path.splitext(self.image_path)
        cut_path = spt_lst[0] + '_cutans' + spt_lst[1]

        region.save(cut_path)

        return cut_path



class EnAnsDetect:
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
        self.edsize = cv2.getStructuringElement(cv2.MORPH_RECT,(8,6))
        #膨胀为提取框
        self.cnts_dilsize = cv2.getStructuringElement(cv2.MORPH_RECT,(5,5))
        #膨胀为提取答案
        self.ans_dilsize = cv2.getStructuringElement(cv2.MORPH_RECT,(6,8))

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

        #输出边缘检测图像
        spt_lst = os.path.splitext(image_path)
        edged_path = spt_lst[0] + '_edged' + spt_lst[1]
        cv2.imwrite(edged_path,self.edged)

        #输出thresh二值图
        thresh_path = spt_lst[0] + '_thresh' + spt_lst[1]
        cv2.imwrite(thresh_path,self.thresh)

    def erode(self,img=None,edsize=None):
        erode_img = cv2.erode(img,edsize)
        #输出腐蚀后的图像
        spt_lst = os.path.splitext(self.image_path)
        erode_path = spt_lst[0] + '_erode' + spt_lst[1]
        cv2.imwrite(erode_path,erode_img)

        return erode_img


    def dilate(self,img=None,dilsize=None):
        dilate_img = cv2.dilate(img,dilsize)
        #输出腐蚀后的图像
        spt_lst = os.path.splitext(self.image_path)
        dil_path = spt_lst[0] + '_dilate' + spt_lst[1]
        cv2.imwrite(dil_path,dilate_img)

        return dilate_img


    def _erode(self):
        '''
        图像腐蚀操作
        '''
        self.erode_image = cv2.erode(self.thresh,self.edsize)
        #输出腐蚀后的图像
        spt_lst = os.path.splitext(self.image_path)
        erode_path = spt_lst[0] + '_erode' + spt_lst[1]
        cv2.imwrite(erode_path,self.erode_image)
        pass


    def _dilate(self,cnts_dilsize=None):
        '''
        图像膨胀操作
        '''
        self.cnts_dilate_image = cv2.erode(self.thresh,self.cnts_dilsize)
        #self.dilate_image = cv2.dilate(self.erode_image,self.edsize1)
        #输出膨胀后的图像
        spt_lst = os.path.splitext(self.image_path)
        dilate_path = spt_lst[0] + '_dilate' + spt_lst[1]
        cv2.imwrite(dilate_path,self.cnts_dilate_image)

    def _erode_dilate(self,dilsize=None):
        '''
        图像膨胀操作
        '''
        #self.dilate_image = cv2.erode(self.thresh,self.edsize)
        self.dilate_image = cv2.dilate(self.erode_image,self.ans_dilsize)
        #输出膨胀后的图像
        spt_lst = os.path.splitext(self.image_path)
        dilate_path = spt_lst[0] + '_dilate' + spt_lst[1]
        cv2.imwrite(dilate_path,self.dilate_image)


    def recKaoHao(self,edsize=None,dilsize=None,dilsize_join=None):
        '''
        识别考号
        '''
        djimg = self.dilate(self.thresh,dilsize_join)
        #腐蚀操作
        eimg = self.erode(djimg,edsize)
        #eimg = self.erode(self.thresh,edsize)
        #膨胀操作
        dilimg = self.dilate(eimg,dilsize)

        #提取考号区域填图的轮廓
        image,cnts,_ = cv2.findContours(dilimg, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        kh_cnts_lst = []
        if len(cnts) > 0:
            cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
            for i,c in enumerate(cnts):
                #轮廓逼近
                peri = cv2.arcLength(c, True)
                approx = cv2.approxPolyDP(c, 0.01 * peri, True)
                (x, y, w, h) = cv2.boundingRect(c)
                kh_cnts_lst.append({'x':x,'y':y,'w':w,'h':h})
            pass
        #去除干扰因素
        from new_detect_image import denoise_data_byh
        if len(kh_cnts_lst)>6:
            kh_cnts_lst =denoise_data_byh(kh_cnts_lst)
        #else:
        #    kh_cnts_lst.append({'h':0,'w':0,'x':0,'y':0})
        #    kh_cnts_lst =denoise_data_byh(kh_cnts_lst)

        #按w去除噪声
        #kh_cnts_lst.append({'h':0,'w':0,'x':0,'y':0})
        #kh_cnts_lst =denoise_data_byw(kh_cnts_lst)
        #if len(kh_cnts_lst)>6:
        #    kh_cnts_lst =denoise_data_byw(kh_cnts_lst)
        #else:
        #    kh_cnts_lst.append({'h':0,'w':0,'x':0,'y':0})
        #    kh_cnts_lst =denoise_data_byw(kh_cnts_lst)

        #mylog(kh_cnts_lst=kh_cnts_lst)

        #考号轮廓按Y排序
        kh_cnts_lst = sorted(kh_cnts_lst,key=lambda x:x['y'])
        #计算填图平均宽度
        avg_w = sum([x['w'] for x in kh_cnts_lst])/len(kh_cnts_lst)

        #计算基准大小
        meta_w = (self.width - avg_w*10)/11

        kh_lst = []
        for _c_x in kh_cnts_lst:
            dis_x = _c_x.get('x')
            kh_no = dis_x/(avg_w+avg_w/3*2)

            kh_no = dis_x/(avg_w+meta_w)
            kh_no = int(round(kh_no))

            kh_lst.append(kh_no)

        return kh_lst

        kh_str = ''.join([str(x) for x in kh_lst])

        #mylog(kh_str=kh_str)
        return kh_str


    def recAns(self,ans_th_rate=[],ans_xx_rate=[],std_quenos=[],edsize=None,dilsize=None,dilsize_join=None):
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

        mylog(std_quenos=std_quenos)


        #设置标准题号坑
        std_ans_queno = []
        std_quenos.sort()
        for i,qn in enumerate(std_quenos):
            dct = {}
            dct['queno'] = str(std_quenos[i])
            dct['rate'] = ans_th_rate[i]
            dct['cnts'] = []
            std_ans_queno.append(dct)

        #是指标准选项标靶
        std_ans_xx = []
        for i,xx in enumerate(ans_xx_rate):
            dct = {}
            dct['rate'] = ans_xx_rate[i] 
            dct["xx"] = char_lst[i] 
            std_ans_xx.append(dct)

        #读取答案轮廓
        djimg = self.dilate(self.thresh,dilsize_join)
        #腐蚀操作
        eimg = self.erode(djimg,edsize)
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

                if y>20 and y>ans_xx_rate[0]*self.height/2:
                    ans_cnts_lst.append({'x':x,'y':y,'w':w,'h':h})


        #获取考号和答案轮廓信息
        #_,ans_cnts_lst = self.findFillCnts(kh_std_dct,edsize,dilsize,dilsize_join)

        #去除干扰因素
        ans_cnts_lst = filter(lambda x:x['h']<self.height/8,ans_cnts_lst)

        ans_cnts_lst.append({'h':0,'w':0,'x':0,'y':0})
        ans_cnts_lst =denoise_data_byh(ans_cnts_lst)

        if len(ans_cnts_lst)>len(std_quenos):
            ans_cnts_lst.append({'h':0,'w':0,'x':0,'y':0})
            #按w去除噪声
            ans_cnts_lst =denoise_data_byw(ans_cnts_lst)

        #把答案轮廓按x排序
        ans_cnts_lst = sorted(ans_cnts_lst,key=lambda x:x['x'])
        #mylog(ans_cnts_lst=ans_cnts_lst)
        
        #取出轮廓里面的最小宽度
        w_lst = [x['w'] for x in ans_cnts_lst]
        w_lst.sort()
        min_w = w_lst[0]

        #相邻X间距小于1/2最小宽度最归结为一组
        import collections
        julei = []
        dct = collections.OrderedDict()
        #ans_cnts_lst.append({'h':0,'w':0,'x':0,'y':0})
        mylog(ans_cnts_lst=ans_cnts_lst)
        for i,_a in enumerate(ans_cnts_lst):
            
            x = _a.get('x')
            if i==0:
                dct[x] = [_a]
            else:
                k = dct.keys()[-1] if dct.keys() else 0

                if x-k < min_w:
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
        #mylog(std_ans_queno=std_ans_queno)

        for i,std_que in enumerate(std_ans_queno):
            #第i题的标准比率
            th_rate = std_que.get('rate')

            #x = julei[i].get('x')
            #w = julei[i].get('w')
            #x = x+w/3
            #_std_th_range = [self.width*th_rate-w/1.5,self.width*th_rate+w/1.5]
            #print u'x标准范围:'
            #print x
            #print _std_th_range
            #continue

            try:

                for j,_jl in enumerate(julei):
                    _cnts = julei[j].get('cnts')

                    x = julei[j].get('x')

                    w = julei[j].get('w')

                    x = x+w/3

                    #第I提x的标准范围
                    _std_th_range = [self.width*th_rate-w/1.6,self.width*th_rate+w/1.6]
                    _std_th_range = [self.width*th_rate-w/1.5,self.width*th_rate+w/1.5]

                    if x>=_std_th_range[0] and x<=_std_th_range[1]:
                        std_ans_queno[i]['cnts'].extend(_cnts)
                        julei.remove(_jl)

            except Exception,e:
                import traceback
                traceback.print_exc()
                pass

        mylog(std_ans_queno=std_ans_queno)


        ################################################
        #将匹配好的题号与轮廓进行处理,解析出选择的答案到底是A,B,C,D
        for que_cnt_dct  in std_ans_queno:

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

                for c in _cnts:
                    ##通过距离在范围内计算
                    y = c.get('y')
                    h = c.get('h')

                    y = y+h/4
                    std_y_ranges = [[self.height*x-avg_h/1.2,self.height*x+avg_h/1.2] for x in ans_xx_rate]
                    std_y_ranges = [[self.height*x-avg_h/1.1,self.height*x+avg_h/1.1] for x in ans_xx_rate]

                    for i,yr in enumerate(std_y_ranges):
                        if y>=yr[0] and y<=yr[1]:
                            que_cnt_dct['ans'].append(char_lst[i])

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


        #mylog(std_ans_queno=std_ans_queno)
        #mylog(shit_lst=shit_lst)

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
    img_path = '010121.jpg'
    img_path = os.path.abspath(img_path)
    paper = Paper(img_path)
    #答案答题卡区域
    kh_area_point,ans_area_point = paper.get_ans_point()
    print ans_area_point
    if ans_area_point:
        kh_img_path = paper.cut_kh_area(kh_area_point)
        ans_img_path = paper.cut_ans_area(ans_area_point)
        
        ##检测和分析答题卡
        enansdetect = EnAnsDetect(kh_img_path)
        #识别考号
        enansdetect.recKaoHao()


