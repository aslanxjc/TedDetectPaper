#coding=utf-8  
import cv2  
import numpy as np    
  
#img = cv2.imread("010102_cut.jpg")  
img = cv2.imread("010121.jpg")  
  
#img = cv2.GaussianBlur(img,(3,3),0)  
#edges = cv2.Canny(img, 50, 150, apertureSize = 3)  

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

thresh = cv2.threshold(gray.copy(),200,255,cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

lines = cv2.HoughLines(thresh,1,np.pi/180,118,100,50)   
result = img.copy()  
for line in lines[0]:  
    rho = line[0] 
    theta= line[1]   
    print rho  
    print theta  
    if  (theta < (np.pi/4. )) or (theta > (3.*np.pi/4.0)):
        pt1 = (int(rho/np.cos(theta)),0)  
        pt2 = (int((rho-result.shape[0]*np.sin(theta))/np.cos(theta)),result.shape[0])  
        cv2.line( result, pt1, pt2, (0,255,0),2)  

        cv2.imwrite('lines.jpg',result)
    else:  
        pt1 = (0,int(rho/np.sin(theta)))  
        pt2 = (result.shape[1], int((rho-result.shape[1]*np.cos(theta))/np.sin(theta)))  
        cv2.line(result, pt1, pt2, (0,255,0), 2) 

        cv2.imwrite('lines.jpg',result)




def _line(path=None):
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
