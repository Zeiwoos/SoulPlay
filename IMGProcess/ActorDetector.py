import cv2
import numpy as np
from IMGProcess.DrawPic import safe_rect

def detect_actor(img, regions):
    """
    检测黄色高亮区域，判断是否为行动者
    """
    h_img, w_img = img.shape[:2]
    
    Yellow_Light_Regions = {}
    IsActor = [False, False, False, False]

    # 预计算 HSV 颜色范围
    lower_bound = np.array([0, 0, 0], dtype=np.uint8)
    upper_bound = np.array([60, 255, 255], dtype=np.uint8)
    kernel = np.ones((11, 11), np.uint8)

    # 直接遍历 regions，提高访问效率
    for idx, (key, region) in enumerate(regions.items()):
        x1, y1, x2, y2 = safe_rect(region["rect"], h_img, w_img)
        hsv = cv2.cvtColor(img[y1:y2, x1:x2], cv2.COLOR_BGR2HSV)

        # 计算黄色遮罩
        mask = cv2.inRange(hsv, lower_bound, upper_bound)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel) 
        
        # 轮廓检测
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            # 直接选取最大轮廓，避免循环
            contour = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(contour)

            if w > 70 or h > 20:
                Yellow_Light_Regions[key] = (x1 + x, y1 + y, w, h)
                IsActor[idx] = True  

    return IsActor, Yellow_Light_Regions
