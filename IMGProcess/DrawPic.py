import cv2
import os

# 安全坐标转换函数
def safe_rect(ratios, h, w):
    x1 = max(0, int(w * ratios[0]))
    y1 = max(0, int(h * ratios[1]))
    x2 = min(w, int(w * ratios[2]))
    y2 = min(h, int(h * ratios[3]))
    return (x1, y1, x2, y2)

def draw_regions(img, hand_regions, regions):
    h_img, w_img = img.shape[:2]
    temp_img = img.copy()
    for key, region in hand_regions.items():
        x1, y1, x2, y2 = safe_rect(region, h_img, w_img)
        if key in regions:
            color = regions[key]['color']
        else:
            color = (255, 255, 255)
        cv2.rectangle(temp_img, (x1, y1), (x2, y2), color, 2)
    temp_img = resize_for_display(temp_img)
    # cv2.imshow('temp_img', temp_img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

def draw_original_regions(img, regions):
    h_img, w_img = img.shape[:2]
    temp_img = img.copy()
    for key, region in regions.items():
        x1, y1, x2, y2 = safe_rect(region['rect'], h_img, w_img)
        cv2.rectangle(temp_img, (x1, y1), (x2, y2), region['color'], 2)
    # temp_img = resize_for_display(temp_img)
    # cv2.imshow('temp_img', temp_img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

def resize_for_display(image, max_width=1000, max_height=800):
    h, w = image.shape[:2]
    scale = min(max_width / w, max_height / h)  # 计算缩放比例
    if scale < 1:  # 仅当图片过大时缩小
        image = cv2.resize(image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    return image


    