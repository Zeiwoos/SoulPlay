import os
import cv2
import json
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache, partial
from IMGProcess import *
from IMGProcess.TileStateGenerater import GameStateGenerator
from IMGProcess.DrawPic import safe_rect
from IMGProcess.FirstSplit import find_all_cards_in_region
from IMGProcess.FinalSplit import process_folder
from IMGProcess.ActorDetector import detect_actor
from IMGProcess.Split import save_cropped_regions
import paddleocr
import threading


# 预加载配置数据
with open("Data/json/profile.json", "r", encoding="utf-8") as f:
    profile = json.load(f)

# 初始化全局配置（线程安全）
PATH_CONFIG = {
    'origin_img_folder': profile['PATH']['TestPath'],
    'ScreenShotPath': profile['PATH']['ScreenShotPath'],
    'first_processed': profile['PATH']['Split_FirstPath'],
    'second_processed': profile['PATH']['Split_FinalPath'],
    'game_state_path': profile['PATH']['BoardStatePath']
}

REGION_CONFIG = {
    'phone': (profile['Regions_Phone'], profile['Yellow_Light_Regions_Phone']),
    'pc': (profile['Regions_PC'], profile['Yellow_Light_Regions_PC'])
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
        self.ocr = init_ocr()  # 单例初始化
        self._ocr_lock = threading.Lock()  # 多线程互斥锁
        self._ocr_warmed_up = False
        self.GameState = None

    def _warm_up_ocr(self):
        """只预热一次"""
        if not self._ocr_warmed_up:
            try:
                dummy_img = np.random.randint(0, 255, (64,64,3), dtype=np.uint8)
                with self._ocr_lock:
                    self.ocr.ocr(dummy_img)
                self._ocr_warmed_up = True
                print("🔥 OCR预热完成")
            except Exception as e:
                print(f"🔥 OCR预热失败: {str(e)}")

    def recognize_word(self, roi: np.ndarray) -> list:
        """OCR识别，线程安全"""
        if roi is None or roi.size == 0:
            print("🆑 空输入数据")
            return []

        if not roi.flags['C_CONTIGUOUS']:
            roi = np.ascontiguousarray(roi)
        if roi.dtype != np.uint8:
            roi = roi.astype(np.uint8)
        if len(roi.shape) == 2:
            roi = cv2.cvtColor(roi, cv2.COLOR_GRAY2RGB)
        elif roi.shape[2] == 4:
            roi = cv2.cvtColor(roi, cv2.COLOR_BGRA2RGB)
        else:
            roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)

        if roi.shape[0] < 5 or roi.shape[1] < 5:
            print(f"📏 忽略过小区域: {roi.shape}")
            return []

        # 模型预热（只执行一次）
        self._warm_up_ocr()

        try:
            with self._ocr_lock:
                results = self.ocr.ocr(roi, det=False, cls=True)
        except Exception as e:
            print(f"❌ OCR异常: {str(e)}")
            return []

        recognized_texts = []
        for res in results:
            if isinstance(res, list):
                for line in res:
                    if isinstance(line, tuple) and len(line) == 2 and isinstance(line[0], str):
                        recognized_texts.append(line[0])
                    else:
                        print(f"⚠️ Unexpected OCR result structure: {line}")

        return recognized_texts if recognized_texts else ""
        
    def process(self, img_path:str)-> bool:
        """处理单个图像的全流程"""
        try:
            # 阶段1：图像读取和基础处理
            img = cv2.imread(img_path)
            if img is None:
                return
            
            h, w = img.shape[:2]
            img_name = os.path.basename(img_path)
            
            # 阶段2：并行处理独立任务
            with ThreadPoolExecutor(max_workers=2) as executor:
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
            game_state_useful = self._save_and_generate(img, hand_regions, img_name, text_self_wind, text_field_wind)

            return game_state_useful

        except Exception as e:
            print(f"处理 {img_path} 失败: {str(e)}")

    def _process_wind(self, img, h, w, wind_type): 
        """风牌识别专用方法（增强校验）"""
        # 获取安全区域
        Wind = safe_rect(self.regions[wind_type]['rect'], h, w)
        
        # 严格校验坐标有效性
        x1, y1, x2, y2 = Wind
        if (x2 <= x1) or (y2 <= y1) or (x1 < 0) or (y1 < 0) or (x2 > w) or (y2 > h):
            print(f"⛔ 无效风牌区域: {Wind}")
            return []
            
        # 提取ROI并复制数据（解决内存对齐问题）
        roi = img[y1:y2, x1:x2].copy()  # 使用copy()避免视图问题
        
        return self.recognize_word(roi)
    
    def _save_and_generate(self, 
                           img:np.ndarray, 
                           regions:dict, 
                           img_name:str, 
                           text_self_wind:list, 
                           text_field_wind:list)-> bool:
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
        print(f"生成游戏状态: {os.path.splitext(img_name)[0]}")
        generator = GameStateGenerator(WindCoding(text_self_wind[0]), 
                                       WindCoding(text_field_wind[0]), 
                                       self.GameState)
        print("Generator初始化完成")
        generator.find_subfolders_with_suffix_scandir(os.path.splitext(img_name)[0])

        game_state_useful = generator.save_board_state(PATH_CONFIG['game_state_path'])

        return game_state_useful

    def update(self, is_phone:bool, GameState:str)-> None:
        """更新配置"""
        self.is_phone = is_phone
        self.regions, self.yellow_regions = REGION_CONFIG['phone' if is_phone else 'pc']
        self.GameState = GameState

def ImageDetection(filePath:str, ImageProcessor:ImageProcessor, GameState:str)-> bool:
    """单图片处理优化"""
    img = cv2.imread(filePath)
    if img is None:
        print(f"⚠️ 无法读取图像: {filePath}")
        return
    h, w = img.shape[:2]
    
    ImageProcessor.update(max(w, h)/min(w, h) > 2, GameState)
    game_state_useful = ImageProcessor.process(filePath)

    return game_state_useful