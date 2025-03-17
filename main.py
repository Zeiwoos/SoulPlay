import os
import cv2
import json
import time
import psutil
import threading
from GameRunStateTest import get_game_state
from IMGProcess.Draw import draw_regions
from IMGProcess.Draw import draw_original_regions
from IMGProcess.Draw import safe_rect
from IMGProcess.FinalSplit import process_folder
from IMGProcess.FirstSplit import save_cropped_regions, recognize_word, find_all_cards_in_region
from IMGProcess.ActorDetector import detect_actor
from ScreenShot.GameShot import GameScreenCapturer
from IMGProcess.StateGenerater import GameStateGenerator

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

def main():
    # 检查路径
    check_path(profile['PATH'])
    check_path(profile['Templates'])
    processImg(ScreenShotPath, first_processed_img_folder, second_processed_img_folder) 

    # 生成游戏状态
    generator = GameStateGenerator()
        
    # 生成并保存游戏状态
    generator.save_game_state(
    output_path=game_state_json_path
    )


def is_game_running(game_name, capturer):
    """检查游戏进程是否在运行，并控制截图"""
    global profile
    game_was_running = False  # 记录上一次游戏是否运行
    while True:
        running = False
        for process in psutil.process_iter(attrs=["pid", "name"]):
            if game_name.lower() in process.info["name"].lower():
                running = True
                break

        profile["is_game_running"] = running

        # **游戏状态变化时，启动或停止截图**
        if running and not game_was_running:
            print(f"🔵 游戏 {game_name} 运行中，开始截图...")
            capturer.start()
        elif not running and game_was_running:
            print(f"🔴 游戏 {game_name} 已关闭，停止截图...")
            capturer.stop()

        game_was_running = running  # 更新状态
        time.sleep(5)  # **每 5 秒检测一次**




# 使用示例
if __name__ == "__main__":
    main()

# if __name__ == '__main__':
#     # 初始化配置
#     capturer = GameScreenCapturer()
    
#     # 自定义配置
#     capturer.configure(
#         interval=ScreenShotInterval,  # 500ms截图间隔
#         output_dir=profile['ScreenShot']['ScreenShotPath']
#     )
#     # 启动游戏检测线程
#     game_monitor_thread = threading.Thread(target=is_game_running(profile["game_name"], capturer), daemon=True)
#     game_monitor_thread.start()
#     # 主线程保持运行，防止进程退出
#     try:
#         while True:
#             time.sleep(1)
#     except KeyboardInterrupt:
#         print("🔴 退出程序...")