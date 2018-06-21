#coding=utf-8  
import cv2  
import numpy as np    
import os
  

def _line(path=None):
    '''
    直线检测
    '''
    lines_data = []
    minLineLength = 200
    maxLineGap = 100
    
    img = cv2.imread(path)  
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    thresh = cv2.threshold(gray.copy(),200,255,cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    #lines = cv2.HoughLinesP(thresh,1,np.pi/180,118,minLineLength,maxLineGap)
    lines = cv2.HoughLinesP(thresh,1,np.pi,118,minLineLength,maxLineGap)
    for line in lines[:,:,:]:
        _tmp_dct = {}
        print line[0,:]
        x1,y1,x2,y2 = line[0,:]
        #print x1,y1,x2,y2
        _tmp_dct["x1"] = x1
        _tmp_dct["y1"] = y1
        _tmp_dct["x2"] = x2
        _tmp_dct["y2"] = y2
        
        lines_data.append(_tmp_dct)

        cv2.line(img,(x1,y1),(x2,y2),(0,255,0),1)
        #输出答题卡区域图像
        ans_lst = os.path.splitext(path)
        ans_path = ans_lst[0] + '_line' + ans_lst[1]
        cv2.imwrite(ans_path,img)
    return lines_data

if __name__ == "__main__":
    lines_data = _line("test_cut_(388, 609, 2188, 998).jpg")
    print lines_data,33333333333333333333
