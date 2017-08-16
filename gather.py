# -*- coding:utf-8 -*-
import numpy as np
import cv2
import matplotlib.pyplot as plt
import time 
import sys

def startserver():
    cap = cv2.VideoCapture(0)
    while(True):
        ret, frame = cap.read() # 读取一帧
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # 转为灰度模式
        cv2.imshow('frame',gray) # 显示帧
        input = chr(cv2.waitKey(1) & 0xFF)
        mark = marksample(input)
        if mark:
            storesample(mark,gray)
        
        if input == "q":
            break
        
    cap.release()
    cv2.destroyAllWindows()
    
def storesample(mark,gray):
    fname = "datas/{}_{}.npy".format(mark,time.time())
    np.save(fname,gray)
    return fname

def marksample(input):
    if input in [str(i) for i in xrange(10)]:
        result = int(input)
    else:
        return None
    return result

if __name__ == "__main__":  
    startserver()


# plt.imshow(frame)
# plt.show()




















