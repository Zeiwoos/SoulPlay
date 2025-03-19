import os
import cv2
import json
import threading
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache, partial
from IMGProcess import *
from IMGProcess.TileStateGenerater import GameStateGenerator
from IMGProcess.DrawPic import safe_rect
from IMGProcess.FirstSplit import recognize_word, find_all_cards_in_region
from IMGProcess.FinalSplit import process_folder
from IMGProcess.ActorDetector import detect_actor
from IMGProcess.Split import save_cropped_regions
import paddleocr


# 预加载配置数据
with open("Data/json/profile.json", "r", encoding="utf-8") as f:
    profile = json.load(f)

# 初始化全局配置（线程安全）
PATH_CONFIG = {
    'origin_img_folder': profile['PATH']['TestPath'],
    'ScreenShotPath': profile['PATH']['ScreenShotPath'],
    'first_processed': profile['PATH']['Split_FirstPath'],
    'second_processed': profile['PATH']['Split_FinalPath'],
    'game_state': profile['PATH']['GameStatePath']
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

class ImageProcessor:
    """图像处理流水线"""
    
    def __init__(self, is_phone):
        self.is_phone = is_phone
        self.regions, self.yellow_regions = REGION_CONFIG['phone' if is_phone else 'pc']
        self.ocr = init_ocr()
        
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
                # 风牌识别
                wind_future = executor.submit(self._process_wind, img, h, w)
                # 行动人检测
                actor_future = executor.submit(detect_actor, img, self.yellow_regions)
                # 区域处理
                region_future = executor.submit(find_all_cards_in_region, img, self.regions)
                
                text_wind = wind_future.result()
                is_actor, yellow = actor_future.result()
                hand_regions = region_future.result()
                print(f"处理 {img_path} 完成\n风牌:{text_wind}，行动人:{is_actor}")
            
            # 阶段3：顺序处理依赖任务
            self._save_and_generate(img, hand_regions, img_name, is_actor, text_wind)
            
        except Exception as e:
            print(f"处理 {img_path} 失败: {str(e)}")
    
    def _process_wind(self, img, h, w):
        """风牌识别专用方法"""
        rect = safe_rect(self.regions['Wind']['rect'], h, w)
        return recognize_word(img[rect[1]:rect[3], rect[0]:rect[2]])
    
    def _save_and_generate(self, img, regions, img_name, is_actor, text_wind):
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
        generator = GameStateGenerator(os.path.splitext(img_name)[0])
        generator.save_game_state(PATH_CONFIG['game_state'])

def batch_processor(img_folder):
    """批量处理优化"""
    # 收集所有待处理文件路径
    file_paths = []
    for root, _, files in os.walk(img_folder):
        file_paths.extend(
            os.path.join(root, f) for f in files 
            if f.lower().endswith(('png', 'jpg', 'jpeg'))
        )
    
    # 按设备类型分组处理
    phone_files = []
    pc_files = []
    for path in file_paths:
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        h, w = img.shape[:2]
        if max(w, h)/min(w, h) > 2:
            phone_files.append(path)
        else:
            pc_files.append(path)
    
    # 并行处理不同设备类型的图片
    with ThreadPoolExecutor(max_workers=2) as device_executor:
        device_executor.submit(_process_batch, phone_files, True)
        device_executor.submit(_process_batch, pc_files, False)

def _process_batch(file_paths, is_phone):
    """批量处理同类型设备图片"""
    processor = ImageProcessor(is_phone)
    with ThreadPoolExecutor(max_workers=os.cpu_count()//2) as executor:
        executor.map(processor.process, file_paths)

def ImageDetection(filename):
    """单图片处理优化"""
    img_path = os.path.join(PATH_CONFIG['ScreenShotPath'], filename)
    img = cv2.imread(img_path)
    h, w = img.shape[:2]
    
    processor = ImageProcessor(max(w, h)/min(w, h) > 2)
    processor.process(img_path)