import os
import cv2
import json
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache, partial
from IMGProcess import *
from IMGProcess.TileStateGenerater import GameStateGenerator
from IMGProcess.DrawPic import safe_rect, draw_regions
from IMGProcess.FirstSplit import recognize_word, find_all_cards_in_region
from IMGProcess.FinalSplit import process_folder
from IMGProcess.ActorDetector import detect_actor
from IMGProcess.Split import save_cropped_regions
import paddleocr
from ActionGenerator import MahjongActionDetector


# 预加载配置数据
with open("Data/json/profile.json", "r", encoding="utf-8") as f:
    profile = json.load(f)

# 初始化全局配置（线程安全）
PATH_CONFIG = {
    'origin_img_folder': profile['PATH']['TestPath'],
    'ScreenShotPath': profile['PATH']['ScreenShotPath'],
    'first_processed': profile['PATH']['Split_FirstPath'],
    'second_processed': profile['PATH']['Split_FinalPath'],
    'game_state_path': profile['PATH']['GameStatePath']
}

REGION_CONFIG = {
    'phone': (profile['regions_phone'], profile['Yellow_Light_Regions_phone']),
    'pc': (profile['regions_pc'], profile['Yellow_Light_Regions_pc'])
}

# 预加载OCR模型（假设recognize_word使用PaddleOCR）
@lru_cache(maxsize=None)
def init_ocr():
    return paddleocr.PaddleOCR(use_angle_cls=True, lang="ch", show_log=False)

# OpenCV优化配置
cv2.setNumThreads(4)

def WindCoding(wind: str) -> str:
    """风牌编码"""
    wind_map = {
        "東": "1z",
        "东": "1z",
        "南": "2z",
        "西": "3z",
        "北": "4z"
    }

    for char in wind_map:
        if char in wind:
            return wind_map[char]
    
    return None  # 如果没有匹配的风向，返回 None

class ImageProcessor:
    """图像处理流水线"""
    def __init__(self):
        self.is_phone = None
        self.regions, self.yellow_regions = None, None
        self.ocr = init_ocr()
        self.GameState = None
        self.MahjongActionDetector = MahjongActionDetector()
        
    def process(self, img_path):
        """处理单个图像的全流程"""
        try:
            # 阶段1：图像读取和基础处理
            img = cv2.imread(img_path)
            if img is None:
                return
            
            h, w = img.shape[:2]
            img_name = os.path.basename(img_path)
            
            # 阶段2：并行处理独立任务
            with ThreadPoolExecutor(max_workers=3) as executor:
                # 字风识别
                self_wind_future = executor.submit(self._process_wind, img, h, w, "Self_Wind")
                field_wind_future = executor.submit(self._process_wind, img, h, w, "Field_Wind")

                # # 行动人检测
                # actor_future = executor.submit(detect_actor, img, self.yellow_regions)

                # 区域处理
                region_future = executor.submit(find_all_cards_in_region, img, self.regions)

                # is_actor, yellow = actor_future.result()
                hand_regions = region_future.result()
                text_self_wind, text_field_wind = self_wind_future.result(), field_wind_future.result()
 
            # 阶段3：顺序处理依赖任务
            print(text_self_wind,text_field_wind)
            self._save_and_generate(img, hand_regions, img_name, text_self_wind, text_field_wind)

        except Exception as e:
            print(f"处理 {img_path} 失败: {str(e)}")

    def _process_wind(self, img, h, w, wind_type): 
        """风牌识别专用方法"""
        Wind = safe_rect(self.regions[wind_type]['rect'], h, w)

        roi = img[Wind[1]:Wind[3], Wind[0]:Wind[2]]

        return recognize_word(roi)
    
    def _save_and_generate(self, img, regions, img_name:str, text_self_wind:list, text_field_wind:list):
        """保存结果并生成游戏状态"""
        # 并行保存操作
        with ThreadPoolExecutor(max_workers=2) as io_executor:
            # 保存第一次分割结果
            first_path = io_executor.submit(
                save_cropped_regions,
                img, regions, img_name, PATH_CONFIG['first_processed']
            ).result()
            
            # 保存第二次分割结果
            io_executor.submit(
                process_folder, first_path, PATH_CONFIG['second_processed']
            )
        
        # 生成游戏状态
        generator = GameStateGenerator(os.path.splitext(img_name)[0], WindCoding(text_self_wind[0]), WindCoding(text_field_wind[0]), self.GameState)

        game_state = generator.save_game_state(PATH_CONFIG['game_state_path'])

        # 生成麻将行动
        self.MahjongActionDetector.process(game_state)

    def update(self, is_phone:bool, GameState:str)-> None:
        """更新配置"""
        self.is_phone = is_phone
        self.regions, self.yellow_regions = REGION_CONFIG['phone' if is_phone else 'pc']
        self.GameState = GameState

def ImageDetection(filename:str, ImageProcessor:ImageProcessor, GameState:str)-> None:
    """单图片处理优化"""
    img_path = os.path.join(PATH_CONFIG['ScreenShotPath'], filename)
    img = cv2.imread(img_path)
    h, w = img.shape[:2]
    
    ImageProcessor.update(max(w, h)/min(w, h) > 2, GameState)
    ImageProcessor.process(img_path)