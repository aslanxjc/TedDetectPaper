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

from detect_image_std import get_std_point

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



class ImagePreTouch:
    """答题卡预处理
    """
    def __init__(self,image_path="test.jpg",std_point=None):
        self.image_path = image_path
        self.image = cv2.imread(self.image_path)

        #图片的宽高度
        self.width = self.image.shape[1]
        self.height = self.image.shape[0]

        #self.std_point = get_std_point(image_path)
        self.std_point = std_point 

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

    def _dilate(self,w=4,h=4):
    #def _dilate(self,w=2,h=4):
        '''
        图像膨胀操作
        '''
        dilsize = cv2.getStructuringElement(cv2.MORPH_RECT,(w,h))
        self.dilate_image = cv2.dilate(self.thresh,dilsize)
        #输出膨胀操作后的图像
        spt_lst = os.path.splitext(self.image_path)
        dilate_path = spt_lst[0] + '_dilate' + spt_lst[1]
        cv2.imwrite(dilate_path,self.dilate_image)
        return dilate_path


    def _inverse(self,w=4,h=4):
        """
        """
        dilate_path = self._dilate(w,h)

        #img1 = 0*np.ones_like(self.gray)
        #img2 = img1-self.gray
        ##输出反色操作后的图像
        spt_lst = os.path.splitext(self.image_path)
        inverse_path = spt_lst[0] + '_dilate_inverse' + spt_lst[1]
        #cv2.imwrite(inverse_path,img2)
        from PIL import Image, ImageOps
        im02 = Image.open()
        im = ImageOps.invert(im02)
        im.save(inverse_path)

        return inverse_path






class CloseImage:

    def __init__(self,image_path="test.jpg"):
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


    def closeopration(self,w=18,h=15):  
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (w, h))  
        self.iClose = cv2.morphologyEx(self.image, cv2.MORPH_CLOSE, kernel)  
        #输出闭操作后的图像
        spt_lst = os.path.splitext(self.image_path)
        close_path = spt_lst[0] + '_close' + spt_lst[1]
        cv2.imwrite(close_path,self.iClose)
        return close_path  




class CloseCntsDetect:

    def __init__(self,image_path="test.jpg"):
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


    def get_cnts(self,std_point=None):
        """识别出答案轮廓
        """

        #cnts = cv2.findContours(self.dilate_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cv2.findContours(self.thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        all_attr = cnts[1]
        cnts = cnts[0] if imutils.is_cv2() else cnts[1]

        all_c = []
        if len(cnts) > 0:
            cnts = sorted(cnts, key=cv2.contourArea, reverse=True)

            for c in cnts:
                peri = 0.01*cv2.arcLength(c, True)

                approx = cv2.approxPolyDP(c, peri, True)
                (x, y, w, h) = cv2.boundingRect(c)

                all_c.append({'x':x,'y':y,'w':w,'h':h})
            #mylog(all_c=all_c)
            all_c = sorted(all_c,key=lambda x:x["w"],reverse=True)
        return all_c


def get_kh_std_dct(path="test_cut.jpg"):
    """获取答题卡考号行
    """
    #膨胀与反色
    image_pre = ImagePreTouch(path)
    inverse_path = image_pre._inverse()

    inverse_image = cv2.imread(inverse_path)

    #图片的宽度
    width = inverse_image.shape[1]
    height = inverse_image.shape[0]

    #图片转成灰度图
    inverse_gray = cv2.cvtColor(inverse_image, cv2.COLOR_BGR2GRAY)
    #对灰度图进行二值化处理
    _,inverse_thresh = cv2.threshold(inverse_gray,0,250,cv2.THRESH_BINARY_INV |\
                     cv2.THRESH_OTSU)

    #识别竖线
    lines_point = []
    minLineLength = 20
    maxLineGap = 10
    #lines = cv2.HoughLinesP(inverse_thresh,1,np.pi/180,118,minLineLength,maxLineGap)
    lines = cv2.HoughLinesP(inverse_thresh,10,np.pi,100,minLineLength,maxLineGap)
    lines1 = lines[:,0,:]
    #for x1,y1,x2,y2 in lines[0]:
    for x1,y1,x2,y2 in lines1[:]:
        if int(y1-y2) > height/4*3:
            lines_point.append({"x1":x1,"y1":y1})
            cv2.line(inverse_image,(x1,y1),(x2,y2),(0,255,0),1)
            #输出答题卡区域图像
            ans_lst = os.path.splitext(path)
            ans_path = ans_lst[0] + '_ansarea_line' + ans_lst[1]
            cv2.imwrite(ans_path,inverse_image)

    lines_point = sorted(lines_point,key=lambda x:x["x1"])
    mylog(lines_point=lines_point)
    #做归并处理
    import collections
    tmp_dct = collections.OrderedDict()
    julei = []
    #tmp_dct = {}
    for _p in lines_point:
        keys_last = tmp_dct.keys()[-1] if tmp_dct.keys() else 0
        x1 = _p.get("x1")
        y1 = _p.get("y1")
        if not tmp_dct.has_key(x1) and x1-keys_last>10:
            tmp_dct[x1] = y1
        else:
            continue
    mylog(tmp_dct=tmp_dct)
    for k,v in tmp_dct.items():
        _dct = {}
        _dct["x1"] = k
        _dct["y1"] = v
        julei.append(_dct)

    mylog(julei=julei)
    if len(julei)==5:
        x1 = julei[-2]["x1"]
        y1 = julei[-2]["y1"]
        x2 = julei[-1]["x1"]
        y2 = julei[-1]["y1"]
        
        kh_std_dct = {"x":x1,"y":y1,"w":x2-x1,"h":0}

        return kh_std_dct

    #raise Exception

    #轮廓提取
    img,cnts,hierarchy = cv2.findContours(inverse_thresh.copy(),cv2.RETR_TREE,\
                    cv2.CHAIN_APPROX_SIMPLE)
    #找出考号区域
    kh_std_lst = []
    if len(cnts) > 0:
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
        
        #mylog(cnts=cnts)
        for i,c in enumerate(cnts):
            #轮廓逼近
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.01 * peri, True)
            (x, y, w, h) = cv2.boundingRect(c)

            if h > height/3:
                continue

            if w < width/8:
                continue


            #标识出识别出的机型轮廓并输出
            cv2.drawContours(inverse_image,c,-1,(255,0,255),3)
            spt_lst = os.path.splitext(path)
            draw_path = spt_lst[0] + '_kh_std_width_draw' + spt_lst[1]
            cv2.imwrite(draw_path,inverse_image)

            #if x > width/3:
            if x > width/4:
                pass
                #continue

            kh_std_lst.append({'x':x,'y':y,'w':w,'h':h})


            mylog(kh_std_lst=kh_std_lst)
            if kh_std_lst:
                kh_std_dct = kh_std_lst[len(kh_std_lst)/2]
            else:
                kh_std_dct = None

            return kh_std_dct

        else:
            return None


def get_ans_area_cnts(path="test_cut.jpg",std_point=None,std_quenos=[]):
    """从答题卡中识别出图像识别技术清洗后的轮廓包含考号和答案
    """
    w = std_point.get("w")
    h = std_point.get("h")
    #膨胀与反色
    if len(std_quenos) > 20:
        _w = 1
        _h = 1
    else:
        #_w = 4
        #_h = 4
        _w = 4
        _h = 2
        #XZB
        #_w = 2
        #_h = 2
        #cs
        #_w = w/3
        #_h = h/3
    image_pre = ImagePreTouch(path,std_point)
    inverse_path = image_pre._inverse(_w,_h)

    #闭操作清除杂质
    image = CloseImage(inverse_path)
    if len(std_quenos) > 20:
        cimage_path = image.closeopration(int(w/3.5),int(h/3.5))
        cimage_path = image.closeopration(int(w/4.5),int(h/3.5))
    else:
        cimage_path = image.closeopration(int(w/1.5),int(h/2.5))
        cimage_path = image.closeopration(int(w/2.5),int(h/3.5))
    #从闭操作后的图像提取轮廓
    ccimage = CloseCntsDetect(cimage_path)
    cnts = ccimage.get_cnts()
    mylog(cnts=cnts)

    return cnts


#############################################
############################################
#英语上下结构
def get_en_kh_std_dct(path="test_cut.jpg"):
    """获取答题卡考号行
    """
    #膨胀与反色
    image_pre = ImagePreTouch(path)
    inverse_path = image_pre._inverse()

    inverse_image = cv2.imread(inverse_path)

    #图片的宽度
    width = inverse_image.shape[1]
    height = inverse_image.shape[0]

    #图片转成灰度图
    inverse_gray = cv2.cvtColor(inverse_image, cv2.COLOR_BGR2GRAY)
    #对灰度图进行二值化处理
    _,inverse_thresh = cv2.threshold(inverse_gray,0,250,cv2.THRESH_BINARY_INV |\
                     cv2.THRESH_OTSU)

    #识别竖线
    lines_point = []
    minLineLength = 20
    maxLineGap = 10
    #lines = cv2.HoughLinesP(inverse_thresh,1,np.pi/180,118,minLineLength,maxLineGap)
    lines = cv2.HoughLinesP(inverse_thresh,10,np.pi,100,minLineLength,maxLineGap)
    lines1 = lines[:,0,:]
    #for x1,y1,x2,y2 in lines[0]:
    for x1,y1,x2,y2 in lines1[:]:
        #if int(y1-y2) > height/4*3:
        if int(y1-y2) > height/3.0:
            lines_point.append({"x1":x1,"y1":y1})
            cv2.line(inverse_image,(x1,y1),(x2,y2),(0,255,0),1)
            #输出答题卡区域图像
            ans_lst = os.path.splitext(path)
            ans_path = ans_lst[0] + '_ansarea_line' + ans_lst[1]
            cv2.imwrite(ans_path,inverse_image)

    lines_point = sorted(lines_point,key=lambda x:x["x1"])
    mylog(lines_point=lines_point)
    #做归并处理
    import collections
    tmp_dct = collections.OrderedDict()
    julei = []
    #tmp_dct = {}
    for _p in lines_point:
        keys_last = tmp_dct.keys()[-1] if tmp_dct.keys() else 0
        x1 = _p.get("x1")
        y1 = _p.get("y1")
        if not tmp_dct.has_key(x1) and x1-keys_last>10:
            tmp_dct[x1] = y1
        else:
            continue
    mylog(tmp_dct=tmp_dct)
    for k,v in tmp_dct.items():
        _dct = {}
        _dct["x1"] = k
        _dct["y1"] = v
        julei.append(_dct)

    mylog(julei=julei)
    if len(julei)==5:
        #x1 = julei[-2]["x1"]
        #y1 = julei[-2]["y1"]
        #x2 = julei[-1]["x1"]
        #y2 = julei[-1]["y1"]

        x1 = julei[-1]["x1"]
        y1 = julei[-1]["y1"]
        
        #kh_std_dct = {"x":x1,"y":y1,"w":x2-x1,"h":0}
        kh_std_dct = {"x":x1,"y":y1,"w":width-x1,"h":height-y1}

        mylog(kh_std_dct=kh_std_dct)

        return kh_std_dct

    raise Exception



def get_en_ans_area_cnts(path="test_cut.jpg",std_point=None,std_quenos=[]):
    """从答题卡中识别出图像识别技术清洗后的轮廓包含考号和答案
    """
    w = std_point.get("w")
    h = std_point.get("h")
    #膨胀与反色
    if len(std_quenos) > 20:
        _w = 1
        _h = 1
    else:
        #_w = 4
        #_h = 4
        _w = 4
        _h = 2
        #XZB
        #_w = 2
        #_h = 2
        #cs
        #_w = w/3
        #_h = h/3
    image_pre = ImagePreTouch(path,std_point)
    inverse_path = image_pre._inverse(_w,_h)

    #闭操作清除杂质
    image = CloseImage(inverse_path)
    if len(std_quenos) > 20:
        cimage_path = image.closeopration(int(w/3.5),int(h/3.5))
        cimage_path = image.closeopration(int(w/4.5),int(h/3.5))
    else:
        cimage_path = image.closeopration(int(w/1.5),int(h/2.5))
        cimage_path = image.closeopration(int(w/2.5),int(h/3.5))
    #从闭操作后的图像提取轮廓
    ccimage = CloseCntsDetect(cimage_path)
    cnts = ccimage.get_cnts()
    mylog(cnts=cnts)

    return cnts




def main():
    std_point = {'y': 102, 'x': 174, 'w': 25, 'h': 32}
    std_quenos = [(1, 0), (2, 0), (3, 0), (4, 0),
                     (1, 0), (2, 0), (3, 0), (4, 0), (5, 0),
                     (6, 0), (7, 0), (8, 0), (9, 0), (10, 0),
                     (11, 0), (12, 0), (13, 0), (14, 0), (15, 0),
                     (16, 0), (17, 0), (18, 0), (19, 0), (20, 0),
                     (16, 0), (17, 0), (18, 0), (19, 0), (20, 0),
                     (16, 0), (17, 0), (18, 0), (19, 0), (20, 0),
                     (16, 0), (17, 0), (18, 0), (19, 0), (20, 0),
                     ]
    #get_kh_std_dct()
    #get_ans_area_cnts()
    #get_en_kh_std_dct()
    print std_point
    get_en_ans_area_cnts("test_cut.jpg",std_point,std_quenos)



if __name__ == "__main__":
    main()
