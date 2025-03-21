import cv2
import os
import numpy as np

def safe_rect(ratios, h, w):
    """安全计算矩形区域"""
    x1 = max(0, int(w * ratios[0]))
    y1 = max(0, int(h * ratios[1]))
    x2 = min(w, int(w * ratios[2]))
    y2 = min(h, int(h * ratios[3]))
    return (x1, y1, x2, y2)

def draw_regions(img:np.ndarray, hand_regions:dict, regions:dict)-> None:
    """绘制hand_regions切割区域"""
    h_img, w_img = img.shape[:2]
    temp_img = img.copy()
    for key, region in hand_regions.items():
        x1, y1, x2, y2 = safe_rect(region, h_img, w_img)
        if key in regions:
            color = regions[key]['Color']
        else:
            color = (255, 255, 255)
        cv2.rectangle(temp_img, (x1, y1), (x2, y2), color, 2)
    temp_img = resize_for_display(temp_img)
    # cv2.imshow('temp_img', temp_img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

def draw_original_regions(img:np.ndarray, regions:dict)-> None:
    """绘制原始切割区域"""
    h_img, w_img = img.shape[:2]
    temp_img = img.copy()
    for key, region in regions.items():
        x1, y1, x2, y2 = safe_rect(region['rect'], h_img, w_img)
        cv2.rectangle(temp_img, (x1, y1), (x2, y2), region['Color'], 2)
    temp_img = resize_for_display(temp_img)
    # cv2.imshow('temp_img', temp_img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

def resize_for_display(image:np.ndarray, max_width=1000, max_height=800):
    """限制图片大小，防止显示过大"""
    h, w = image.shape[:2]
    scale = min(max_width / w, max_height / h)  # 计算缩放比例
    if scale < 1:  # 仅当图片过大时缩小
        image = cv2.resize(image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    return image  