#-*-coding:utf-8 -*-
def denoise_data_byw(lst):
    '''
    算法去噪声
    lst = [{'h':29,'w':110},{'h':26,'w':110},{'h':24,'w':110},
    {'h':2,'w':110},{'h':2,'w':110},{'h':3,'w':110}]
    '''
    lst = sorted(lst,key=lambda x:x['w'],reverse=True)
    #mylog(lst=lst)
    #print 6666666666666666666666666
    h_lst = [x['w'] for x in lst]

    #print h_lst,77777777777777777
    ##计算相邻两个元素的差值
    dis_lst = []
    for i,v in enumerate(h_lst):
        if i < len(h_lst)-2:
            dis = h_lst[i]-h_lst[i+1]
            dis_lst.append(dis)
    #print dis_lst,888888888888888
    #org_dis_lst = copy.deepcopy(dis_lst)
    dis_lst.sort()
    dis_lst.reverse()

    ##相邻差值倒序排列

    #print dis_lst,4444444444444444
    #print dis_lst[0]

    #print type(dis_lst)
    #dis_max_1 = dis_lst[0]
    #print dis_max,3333333333333333
    #index = org_dis_lst.index(dis_max)
    #print lst,1212121

    #lst = lst[:index+1]

    return lst
    
    
a=[{'h': 57, 'w': 111, 'x': 1763, 'y': 158},
 {'h': 50, 'w': 110, 'x': 1882, 'y': 157},
 {'h': 48, 'w': 107, 'x': 2110, 'y': 151},
 {'h': 55, 'w': 104, 'x': 2283, 'y': 147},
 {'h': 47, 'w': 104, 'x': 1650, 'y': 162},
 {'h': 58, 'w': 102, 'x': 2003, 'y': 150},
 {'h': 55, 'w': 100, 'x': 2397, 'y': 138},
 {'h': 58, 'w': 96, 'x': 1663, 'y': 539},
 {'h': 50, 'w': 94, 'x': 1779, 'y': 536},
 {'h': 27, 'w': 11, 'x': 2530, 'y': 478},
 {'h': 35, 'w': 10, 'x': 2524, 'y': 106},
 {'h': 40, 'w': 9, 'x': 2523, 'y': 43},
 {'h': 27, 'w': 9, 'x': 2528, 'y': 294}] 
 
denoise_data_byw(a)
