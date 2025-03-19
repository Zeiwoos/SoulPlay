import cv2
import numpy as np
import paddleocr
import os
import logging
from IMGProcess.DrawPic import safe_rect
# 屏蔽 PaddleOCR 的调试信息
logging.getLogger("ppocr").setLevel(logging.ERROR)

def recognize_word(roi):
    ocr = paddleocr.PaddleOCR(use_angle_cls=True, lang="ch", show_log=False)  # 关闭日志
    results = ocr.ocr(roi, det=False, cls=True)

    recognized_texts = []
    for res in results:
        if not isinstance(res, list):  # 避免 res 不是列表的情况
            continue
        for line in res:
            text = line[0][0]  # 获取文字
            recognized_texts.append(text)

    return recognized_texts

def find_all_cards_in_region(img, regions):
    h_img, w_img = img.shape[:2]
    hand_regions = []
    
    for key, region in regions.items():
        x1, y1, x2, y2 = safe_rect(region['rect'], h_img, w_img)
        roi = img[y1:y2, x1:x2]
        
        # 转换为 HSV 颜色空间
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        
        if key == 'Wind':
            continue

        lower_bound = np.array([0, 0, 180])
        upper_bound = np.array([180, 60, 255])
            
        # 颜色过滤
        mask = cv2.inRange(hsv, lower_bound, upper_bound)
            
        kernel_size = 17 if key == 'Hand_Tiles' else 3
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.dilate(mask, kernel, iterations=5 if key != 'Hand_Tiles' else 4)

        # 轮廓检测
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # 筛选麻将牌区域
        if len(contours) > 0 :
            if key == 'Hand_Tiles':
                # 选出最左侧的轮廓
                selected_contour = min(contours, key=lambda c: cv2.boundingRect(c)[0])
            elif key == 'Self_Mingpai':
                # 选出最右侧的轮廓，并确保其Self_Mingpai左边界 >= Hand_Tiles右边界
                selected_contour = max(contours, key=lambda c: cv2.boundingRect(c)[0])
                x, _, w, _ = cv2.boundingRect(selected_contour)
                if x + x1 < hand_regions[0][1] + hand_regions[0][3] : # 确保Self_Mingpai左边界 >= Hand_Tiles右边界
                    selected_contour = None
            else:
                # 根据不同的 `key` 进行筛选
                if key == 'Second_Mingpai':
                    contours = [c for c in contours if (cv2.boundingRect(c)[1] + y1) < (0.15 * h_img)] # 上边界
                elif key == 'Third_Mingpai':
                    contours = [c for c in contours if (cv2.boundingRect(c)[0] + x1) < (0.3 * w_img)] # 左边界
                elif key == 'Fourth_Mingpai':
                    contours = [c for c in contours if (cv2.boundingRect(c)[1] + y1 + cv2.boundingRect(c)[3]) > (0.85 * h_img)  # 下边界大于0.85*h_img
                                and cv2.boundingRect(c)[0] + x1 < hand_regions[0][1]] # Fourth_Mingpai左边界小于Hand_Tiles的左边界
                
                selected_contour = max(contours, key=cv2.contourArea, default=None) if contours else None

            # 如果轮廓不为空，则获取轮廓的边界矩形
            if selected_contour is not None:
                x, y, w, h = cv2.boundingRect(selected_contour)
                if w > 25 and h > 25:
                    hand_regions.append((key, x + x1, y + y1, w, h))
    print(hand_regions)
    return hand_regions

# 保存切割的区域
def save_cropped_regions(img, hand_regions, img_name, output_folder):
    """
    根据检测到的麻将牌区域，裁剪并保存图片
    """
    img_base_name = os.path.splitext(img_name)[0]  # 去掉后缀
    img_output_path = os.path.join(output_folder, img_base_name)
    format = img_name.split('.')[-1]
    name = img_name.split('.')[0]

    # 确保输出文件夹存在
    os.makedirs(img_output_path, exist_ok=True)

    h_img, w_img = img.shape[:2]

    for key, x, y, w, h in hand_regions:
        cropped_img = img[max(0, y-20):min(h_img, y+h+20), max(0, x-20):min(w_img, x+w+20)]  # 裁剪区域
        # 获取格式和文件名
        save_path = os.path.join(img_output_path, f"{name}_{key}.{format}")  # 保存路径

        cv2.imwrite(save_path, cropped_img)  # 保存图片
        print(f"已保存: {save_path}")

    return img_output_path