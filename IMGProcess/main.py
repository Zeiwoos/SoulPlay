from FinalSplit import process_folder
from FirstSplit import save_cropped_regions, recognize_wind, find_all_cards_in_region, safe_rect
from Draw import draw_regions, draw_original_regions
import os
import cv2

origin_img_folder = 'Data/recogition/IMG'
first_processed_img_folder = 'Data/recogition/output_first/'
second_processed_img_folder = 'Data/recogition/output_final/'

def processImg(img_folder, first_processed_img_folder, second_processed_img_folder):
    """
    处理图片
    """
    # 文件夹是否存在
    if not os.path.exists(img_folder):
        print(f"Error: {img_folder} does not exist")
        exit()
    
    # 获取文件夹(Phone/PC)中的所有图片（包含子文件以及子文件中的图片）
    for root, _, files in os.walk(img_folder):
        for file in files:
            if file.lower().endswith(('png', 'jpg', 'jpeg')):
                img_path = os.path.join(root, file)
                img_name = os.path.basename(img_path)
                img = cv2.imread(img_path)

                # 获取图片的宽高
                h, w = img.shape[:2]

                # 定义区域
                regions_phone = {
                    # 手牌区域
                    'Hand_Tiles': { 'rect': safe_rect((0.18, 0.82, 1, 1.0), h, w), 'color': (0, 0, 255) },# 蓝色
                    # 明牌区域
                    'Self_Mingpai': {'rect': safe_rect((0.12, 0.82, 1, 1.0), h, w), 'color': (0, 255, 255) },# 青色
                    'Second_Mingpai': {'rect': safe_rect((0.70, 0.03, 0.89, 0.6), h, w), 'color': (255, 140, 0) },# 深橙色
                    'Third_Mingpai': {'rect': safe_rect((0.20, 0, 0.60, 0.10), h, w), 'color': (255, 0, 255) },# 洋红色
                    'Fourth_Mingpai': {'rect': safe_rect((0.03, 0.3, 0.25, 0.88), h, w), 'color': (128, 0, 128) },# 紫色
                    # 弃牌区域
                    'Self_Discard': {'rect': safe_rect((0.39, 0.497, 0.63, 0.70), h, w), 'color': (0, 255, 0) },# 绿色
                    'Second_Discard': {'rect': safe_rect((0.57, 0.18, 0.75, 0.50), h, w), 'color': (0, 128, 255) },# 天蓝色
                    'Third_Discard': {'rect': safe_rect((0.37, 0.115, 0.62, 0.27), h, w), 'color': (255, 0, 0) },# 红色
                    'Fourth_Discard': {'rect': safe_rect((0.22, 0.12, 0.42, 0.55), h, w), 'color': (128, 128, 0) },# 橄榄绿
                    # 宝牌指示牌
                    'Dora_Indicator': {'rect': safe_rect((0, 0.02, 0.12, 0.12), h, w), 'color': (255, 255, 0) },# 黄色
                    # 风位
                    'Wind': {'rect': safe_rect((0.43, 0.453, 0.458, 0.498), h, w), 'color': (75, 0, 130) },# 靛蓝色
                }
                regions_pc = {
                    'Hand_Tiles': { 'rect': safe_rect((0.11, 0.82, 1, 1.0), h, w), 'color': (0, 0, 255) },# 蓝色
                    # 明牌区域
                    'Self_Mingpai': {'rect': safe_rect((0.12, 0.82, 1, 1.0), h, w), 'color': (0, 255, 255) },# 青色
                    'Second_Mingpai': {'rect': safe_rect((0.70, 0.03, 0.89, 0.6), h, w), 'color': (255, 140, 0) },# 深橙色
                    'Third_Mingpai': {'rect': safe_rect((0.20, 0, 0.60, 0.10), h, w), 'color': (255, 0, 255) },# 洋红色
                    'Fourth_Mingpai': {'rect': safe_rect((0.03, 0.3, 0.25, 0.88), h, w), 'color': (128, 0, 128) },# 紫色
                    # 弃牌区域
                    'Self_Discard': {'rect': safe_rect((0.39, 0.497, 0.63, 0.70), h, w), 'color': (0, 255, 0) },# 绿色
                    'Second_Discard': {'rect': safe_rect((0.57, 0.18, 0.75, 0.50), h, w), 'color': (0, 128, 255) },# 天蓝色
                    'Third_Discard': {'rect': safe_rect((0.37, 0.115, 0.62, 0.27), h, w), 'color': (255, 0, 0) },# 红色
                    'Fourth_Discard': {'rect': safe_rect((0.22, 0.12, 0.42, 0.55), h, w), 'color': (128, 128, 0) },# 橄榄绿
                    # 宝牌指示牌
                    'Dora_Indicator': {'rect': safe_rect((0, 0.02, 0.12, 0.12), h, w), 'color': (255, 255, 0) },# 黄色
                    # 风位
                    'Wind': {'rect': safe_rect((0.42, 0.455, 0.46, 0.50), h, w), 'color': (75, 0, 130) },# 靛蓝色
                }

                # 长宽比
                IsPhone = False
                if max(w, h)/min(w, h) > 2:
                    IsPhone = True
                if IsPhone:
                    regions = regions_phone
                else:
                    regions = regions_pc

                # 使用find_all_cards_in_region函数进行处理
                hand_regions = find_all_cards_in_region(img, regions)
                draw_regions(img, hand_regions, regions)
                draw_original_regions(img, regions)
                # 使用save_cropped_regions函数进行处理
                img_output_path = save_cropped_regions(img, hand_regions, img_name, f"{first_processed_img_folder}{'Phone' if IsPhone else 'PC'}{'/'}")

                # 使用process_folder函数进行处理
                process_folder(img_output_path, f"{second_processed_img_folder}{'Phone' if IsPhone else 'PC'}{'/'}")

                # 识别风牌
                text_wind = recognize_wind(img[regions['Wind']['rect'][1]:regions['Wind']['rect'][3], regions['Wind']['rect'][0]:regions['Wind']['rect'][2]])
                print(text_wind)



if __name__ == '__main__':
    processImg(origin_img_folder, first_processed_img_folder, second_processed_img_folder)
