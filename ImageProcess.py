import os
import cv2
import json
from IMGProcess.DrawPic import draw_regions
from IMGProcess.DrawPic import draw_original_regions
from IMGProcess.DrawPic import safe_rect
from IMGProcess.FinalSplit import process_folder
from IMGProcess.FirstSplit import save_cropped_regions, recognize_word, find_all_cards_in_region
from IMGProcess.ActorDetector import detect_actor
from IMGProcess.TileStateGenerater import GameStateGenerator

global profile
with open("Data/json/profile.json", "r", encoding="utf-8") as f:
    profile = json.load(f)

# 测试图片
origin_img_folder = profile['PATH']['TestPath']
# 截图路径
ScreenShotPath = profile['PATH']['ScreenShotPath']
# 截图间隔
ScreenShotInterval = profile['ScreenShotInterval']
# 第一次分割
first_processed_img_folder = profile['PATH']['Split_FirstPath']
# 第二次分割
second_processed_img_folder = profile['PATH']['Split_FinalPath']
# 游戏状态
game_state_json_path = profile['PATH']['GameStatePath']
# 手机区域
regions_phone = profile['regions_phone']
# PC区域
regions_pc = profile['regions_pc']
# 手机黄光区域
Yellow_Light_Regions_phone = profile['Yellow_Light_Regions_phone']
# PC黄光区域
Yellow_Light_Regions_pc = profile['Yellow_Light_Regions_pc']

# 检测路径是否存在
def check_path(paths):
    for key, path in paths.items():
        if not os.path.exists(path):
            print(f"Error: {path} does not exist")
            # 创建路径
            os.makedirs(path, exist_ok=True)
        else:
            print(f"Path {path} exists")
    return True

def processImgs(img_folder, first_processed_img_folder, second_processed_img_folder):
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
                print(f"-- -- -- -- -- --start Processing {file}-- -- -- -- --")
                img_path = os.path.join(root, file)
                img_name = os.path.basename(img_path)
                img = cv2.imread(img_path)

                # 获取图片的宽高
                h, w = img.shape[:2]

                # 长宽比
                IsPhone = True if max(w, h)/min(w, h) > 2 else False
                regions = regions_phone if IsPhone else regions_pc

                # 识别风牌
                rect = safe_rect(regions['Wind']['rect'], h, w)
                text_wind = recognize_word(img[rect[1]:rect[3], rect[0]:rect[2]])
                print(text_wind)

                IsActor, Yellow_Light_Regions = detect_actor(img, Yellow_Light_Regions_phone if IsPhone else Yellow_Light_Regions_pc)
                print(IsActor)
                draw_regions(img, Yellow_Light_Regions, Yellow_Light_Regions_phone if IsPhone else Yellow_Light_Regions_pc)

                # 使用find_all_cards_in_region函数进行处理
                hand_regions = find_all_cards_in_region(img, regions)
                draw_regions(img, hand_regions, regions)
                draw_original_regions(img, regions)
                # 使用save_cropped_regions函数进行处理
                img_output_path = save_cropped_regions(img, hand_regions, img_name, f"{first_processed_img_folder}{'/'}")

                # 使用process_folder函数进行处理
                process_folder(img_output_path, f"{second_processed_img_folder}{'/'}")

                print(f"-- -- -- -- -- --Processing {file} done-- -- -- -- --\n")

def processSingleImg(img_name, first_processed_img_folder, second_processed_img_folder):
    """
    处理图片
    """
    # 文件是否存在
    img_path = os.path.join(ScreenShotPath, img_name)
    if not os.path.exists(img_path):
        print(f"Error: {img_path} does not exist")
        exit()

    print(f"-- -- -- -- -- --start Processing {img_name}-- -- -- -- --")
    img = cv2.imread(img_path)
    # cv2.imshow("img", img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows

    # 获取图片的宽高
    h, w = img.shape[:2]
    print(f"图片尺寸:高{h} 宽：{w}")

    # 长宽比
    IsPhone = True if max(w, h)/min(w, h) > 2 else False
    print(f"IsPhone: {IsPhone}")
    regions = regions_phone if IsPhone else regions_pc

    # 识别风牌
    rect = safe_rect(regions['Wind']['rect'], h, w)
    text_wind = recognize_word(img[rect[1]:rect[3], rect[0]:rect[2]])
    print(text_wind)

    IsActor, Yellow_Light_Regions = detect_actor(img, Yellow_Light_Regions_phone if IsPhone else Yellow_Light_Regions_pc)
    print(IsActor)
    draw_regions(img, Yellow_Light_Regions, Yellow_Light_Regions_phone if IsPhone else Yellow_Light_Regions_pc)

    # 使用find_all_cards_in_region函数进行处理
    hand_regions = find_all_cards_in_region(img, regions)
    draw_regions(img, hand_regions, regions)
    draw_original_regions(img, regions)
    # 使用save_cropped_regions函数进行处理
    img_output_path = save_cropped_regions(img, hand_regions, img_name, f"{first_processed_img_folder}{'/'}")

    # 使用process_folder函数进行处理
    process_folder(img_output_path, f"{second_processed_img_folder}{'/'}")

    print(f"-- -- -- -- -- --Processing {img_name} done-- -- -- -- --\n")

def ImageDetection(filename):
    """
    图片处理
    """
    # 处理图片
    processSingleImg(filename, first_processed_img_folder, second_processed_img_folder) 

    name = os.path.splitext(filename)[0]
    # 生成游戏状态
    generator = GameStateGenerator(name)
        
    # 生成并保存游戏状态
    generator.save_game_state(
        output_path=game_state_json_path
    )