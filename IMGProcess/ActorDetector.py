import cv2
import numpy as np
from IMGProcess.Draw import safe_rect

def detect_actor(img, regions):
    Yellow_Light_Regions = []
    h_img, w_img = img.shape[:2]
    # cv2.imshow('img', img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # 四人数组
    IsActor = [False, False, False, False]
    for key, region in regions.items():
        x1, y1, x2, y2 = safe_rect(region["rect"], h_img, w_img)
        roi = img[y1:y2, x1:x2]
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        # 暗黄到明黄的HSV范围
        lower_bound = np.array([0, 0, 0])  # 暗黄色
        upper_bound = np.array([60, 255, 255])  # 明黄色

        mask = cv2.inRange(hsv, lower_bound, upper_bound)

        # 腐蚀再恢复
        kernel = np.ones((11, 11), np.uint8)
        mask = cv2.erode(mask, kernel, iterations=1)
        mask = cv2.dilate(mask, kernel, iterations=1)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if w > 70 or h > 20:
                Yellow_Light_Regions.append((key, x + x1, y + y1, w, h))
                if key == 'Self_Yellow_Light':
                    IsActor[0] = True
                elif key == 'Second_Yellow_Light':
                    IsActor[1] = True
                elif key == 'Third_Yellow_Light':
                    IsActor[2] = True
                elif key == 'Fourth_Yellow_Light':
                    IsActor[3] = True

    return IsActor, Yellow_Light_Regions


    



