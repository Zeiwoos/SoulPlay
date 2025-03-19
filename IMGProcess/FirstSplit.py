import cv2
import numpy as np
import paddleocr
import os
import logging
from IMGProcess.DrawPic import safe_rect
cv2.setNumThreads(4)

# 预加载 OCR 模型避免重复初始化
logging.getLogger("ppocr").setLevel(logging.ERROR)
ocr_instance = paddleocr.PaddleOCR(use_angle_cls=True, lang="ch", show_log=False)

def recognize_word(roi):
    results = ocr_instance.ocr(roi, det=False, cls=True)
    return [line[0][0] for res in results if isinstance(res, list) for line in res]

def find_all_cards_in_region(img, regions):
    h_img, w_img = img.shape[:2]
    hand_regions = {}
    kernel_cache = {
        'Hand_Tiles': np.ones((17, 17), np.uint8),
        'default': np.ones((3, 3), np.uint8)
    }
    
    # 预处理颜色范围
    lower = np.array([0, 0, 180])
    upper = np.array([180, 60, 255])
    
    # 优先处理 Hand_Tiles 区域
    if 'Hand_Tiles' in regions:
        key = 'Hand_Tiles'
        region = regions[key]
        x1, y1, x2, y2 = safe_rect(region['rect'], h_img, w_img)
        roi = img[y1:y2, x1:x2]
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, lower, upper)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_cache[key])
        mask = cv2.dilate(mask, kernel_cache[key], iterations=4)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) > 0:
            selected = min(contours, key=lambda c: cv2.boundingRect(c)[0])
            x, y, w, h = cv2.boundingRect(selected)
            if w > 25 and h > 25:
                hand_regions[key] = (x + x1, y + y1, w, h)
    
    # 并行处理其他区域
    from concurrent.futures import ThreadPoolExecutor
    def process_region(key):
        if key in ['Wind', 'Hand_Tiles']:
            return
        region = regions[key]
        x1, y1, x2, y2 = safe_rect(region['rect'], h_img, w_img)
        roi = img[y1:y2, x1:x2]
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, lower, upper)
        kernel = kernel_cache['default']
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.dilate(mask, kernel, iterations=5)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return
        
        selected = None
        if key == 'Self_Mingpai':
            hand_info = hand_regions.get('Hand_Tiles')
            if hand_info:
                right_bound = hand_info[0] + hand_info[2]
                candidates = [c for c in contours if (cv2.boundingRect(c)[0] + x1) >= right_bound]
                selected = max(candidates, key=lambda c: cv2.boundingRect(c)[0]) if candidates else None
        elif key == 'Second_Mingpai':
            candidates = [c for c in contours if (cv2.boundingRect(c)[1] + y1) < (0.15 * h_img)]
            selected = max(candidates, key=cv2.contourArea) if candidates else None
        elif key == 'Third_Mingpai':
            candidates = [c for c in contours if (cv2.boundingRect(c)[0] + x1) < (0.3 * w_img)]
            selected = max(candidates, key=cv2.contourArea) if candidates else None
        elif key == 'Fourth_Mingpai':
            candidates = [c for c in contours if (cv2.boundingRect(c)[1] + y1 + cv2.boundingRect(c)[3]) > (0.85 * h_img)
                          and cv2.boundingRect(c)[0] + x1 < hand_regions.get('Hand_Tiles', (0, 0, 0, 0))[0]]
            selected = max(candidates, key=cv2.contourArea) if candidates else None
        else:
            selected = max(contours, key=cv2.contourArea
                           , default=None)
        
        if selected is not None:
            x, y, w, h = cv2.boundingRect(selected)
            if w > 25 and h > 25:
                return (key, (x + x1, y + y1, w, h))
        return None
    
    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(process_region, key) for key in regions if key not in ['Wind', 'Hand_Tiles']]
        for future in futures:
            result = future.result()
            if result:
                hand_regions[result[0]] = result[1]
    return hand_regions