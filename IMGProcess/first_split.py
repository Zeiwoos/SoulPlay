import cv2
import numpy as np
import paddleocr
import os

# 图片文件夹
img_folder = "Data\\recogition\\IMG"
output_folder = "Data\\recogition\\output_first"

def find_all_cards_in_region(img, regions):
    h_img, w_img = img.shape[:2]
    hand_regions = []
    
    for key, region in regions.items():
        print(key)
        x1, y1, x2, y2 = region['rect']
        roi = img[y1:y2, x1:x2]
        
        # 转换为 HSV 颜色空间
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        
        if key == 'wind':
            # 识别风牌
            ocr = paddleocr.PaddleOCR(use_angle_cls=True, lang="ch")
            results = ocr.ocr(roi, det=False, cls=True)
            for res in results:
                for line in res:
                    print(line)
            continue

        lower_bound = np.array([0, 0, 180])
        upper_bound = np.array([180, 60, 255])
            
        # 颜色过滤
        mask = cv2.inRange(hsv, lower_bound, upper_bound)
            
        kernel_size = 9 if key == 'hand_tiles' else 3
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.dilate(mask, kernel, iterations=3 if key != 'hand_tiles' else 5)

        # 轮廓检测
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # print(len(contours))

        # 筛选麻将牌区域
        if len(contours) > 0 :
            if key == 'hand_tiles':
                # 选出最左侧的轮廓
                selected_contour = min(contours, key=lambda c: cv2.boundingRect(c)[0])
            elif key == 'self_Mingpai':
                # 选出最右侧的轮廓，并确保其右边界 >= 手牌右边界
                selected_contour = max(contours, key=lambda c: cv2.boundingRect(c)[0])
                x, _, w, _ = cv2.boundingRect(selected_contour)
                if x + x1 + w < hand_regions[0][1] + hand_regions[0][3]:
                    selected_contour = None
            else:
                # 根据不同的 `key` 进行筛选
                if key == 'second_Mingpai':
                    contours = [c for c in contours if (cv2.boundingRect(c)[1] + y1) < (0.15 * h_img)]
                elif key == 'third_Mingpai':
                    contours = [c for c in contours if (cv2.boundingRect(c)[0] + x1) < (0.3 * w_img)]
                elif key == 'fourth_Mingpai':
                    contours = [c for c in contours if (cv2.boundingRect(c)[1] + y1 + cv2.boundingRect(c)[3]) > (0.85 * h_img)]
                
                selected_contour = max(contours, key=cv2.contourArea, default=None) if contours else None
            
            if selected_contour is not None:
                x, y, w, h = cv2.boundingRect(selected_contour)
                if w > 25 and h > 25:
                    hand_regions.append((key, x + x1, y + y1, w, h))

    return hand_regions

def draw_regions(img, hand_regions, regions):
    for (key, x, y, w, h) in hand_regions:
        if key in regions:  # 确保 key 存在于 regions
            color = regions[key]['color']
        else:
            color = (255, 255, 255)  # 如果找不到，默认白色
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
        print(key)

# 缩放图片适应屏幕
def resize_for_display(image, max_width=1000, max_height=800):
    h, w = image.shape[:2]
    scale = min(max_width / w, max_height / h)  # 计算缩放比例
    if scale < 1:  # 仅当图片过大时缩小
        image = cv2.resize(image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    return image

# 安全坐标转换函数
def safe_rect(ratios, h, w):
    x1 = max(0, int(w * ratios[0]))
    y1 = max(0, int(h * ratios[1]))
    x2 = min(w, int(w * ratios[2]))
    y2 = min(h, int(h * ratios[3]))
    return (x1, y1, x2, y2)

# 保存切割的区域
def save_cropped_regions(img, hand_regions, img_name, h_img, w_img):
    """
    根据检测到的麻将牌区域，裁剪并保存图片
    """
    img_base_name = os.path.splitext(img_name)[0]  # 去掉后缀
    img_output_path = os.path.join(output_folder, img_base_name)

    # 确保输出文件夹存在
    os.makedirs(img_output_path, exist_ok=True)

    for key, x, y, w, h in hand_regions:
        cropped_img = img[max(0, y-20):min(h_img, y+h+20), max(0, x-20):min(w_img, x+w+20)]  # 裁剪区域
        save_path = os.path.join(img_output_path, f"{key}_{img_name}")  # 保存路径
        cv2.imwrite(save_path, cropped_img)  # 保存图片
        print(f"已保存: {save_path}")

if __name__ == "__main__":
    for file in os.listdir(img_folder):
        img = cv2.imread(os.path.join(img_folder, file))
        h, w = img.shape[:2]

        regions = {
            # 手牌区域
            'hand_tiles': { 'rect': safe_rect((0.12, 0.82, 1, 1.0), h, w), 'color': (0, 0, 255) },# 蓝色
            # 明牌区域
            'self_Mingpai': {'rect': safe_rect((0.12, 0.82, 1, 1.0), h, w), 'color': (0, 255, 255) },# 青色
            'second_Mingpai': {'rect': safe_rect((0.75, 0.05, 0.89, 0.6), h, w), 'color': (255, 140, 0) },# 深橙色
            'third_Mingpai': {'rect': safe_rect((0.20, 0, 0.60, 0.09), h, w), 'color': (255, 0, 255) },# 洋红色
            'fourth_Mingpai': {'rect': safe_rect((0.03, 0.3, 0.18, 0.905), h, w), 'color': (128, 0, 128) },# 紫色
            # 弃牌区域
            'self_discard': {'rect': safe_rect((0.39, 0.497, 0.63, 0.70), h, w), 'color': (0, 255, 0) },# 绿色
            'Second_discard': {'rect': safe_rect((0.57, 0.18, 0.75, 0.50), h, w), 'color': (0, 128, 255) },# 天蓝色
            'third_discard': {'rect': safe_rect((0.37, 0.115, 0.62, 0.27), h, w), 'color': (255, 0, 0) },# 红色
            'fourth_discard': {'rect': safe_rect((0.22, 0.12, 0.42, 0.55), h, w), 'color': (128, 128, 0) },# 橄榄绿
            # 宝牌指示牌
            'dora_indicator': {'rect': safe_rect((0, 0.02, 0.18, 0.12), h, w), 'color': (255, 255, 0) },# 黄色
            # 风位
            'wind': {'rect': safe_rect((0.42, 0.455, 0.46, 0.50), h, w), 'color': (75, 0, 130) },# 靛蓝色
        }
        
        hand_regions = find_all_cards_in_region(img, regions)
        # print("检测到的麻将牌区域:", hand_regions)

        save_cropped_regions(img, hand_regions, file, h, w)

        # # 绘制检测结果
        # result_img = img.copy()
        # draw_regions(result_img, hand_regions, regions)

        # # 调整图片大小以适应显示
        # display_img = resize_for_display(result_img)

        # # 显示结果
        # cv2.imshow("Detected Mahjong Tiles", display_img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
