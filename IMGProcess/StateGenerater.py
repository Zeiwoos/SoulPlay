import os
import json
import cv2
import numpy as np
import shutil
from tqdm import tqdm
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict
from IMGProcess.BatchClassify import BatchClassifier
import json

with open("Data/json/profile.json", "r", encoding="utf-8") as f:
    profile = json.load(f)

def find_subfolders_with_suffix_scandir(parent_folder):
    """使用 os.scandir() 查找一级子文件夹，性能更优"""
    suffix = profile['Suffix']['Suffix']
    folder_list = {}
    for suffix in suffix:
        # 返回 suffix , 文件名的dict
        for entry in os.scandir(parent_folder):
            if entry.is_dir() and entry.name.endswith(suffix):
                folder_list[suffix] = entry.name
    return folder_list

class GameStateGenerator(BatchClassifier):
    """
    游戏状态生成器
    """
    def __init__(self):
        super().__init__()
        self.folder_list = find_subfolders_with_suffix_scandir(profile['PATH']['Split_FinalPath'])

    def process_tiles(self) -> List[str]:
        """多线程处理麻将图片"""
        valid_tiles = {}
        temp_tiles = []
        for key, folder in self.folder_list.items():
            if key == "Dora_Indicator":
                continue
            print(key,folder)
            tile_images = f"{profile['PATH']['Split_FinalPath']}/{folder}"
            print(tile_images)
            # 多线程处理
            futures = []
            with ThreadPoolExecutor(max_workers=4) as executor:
                for path in os.listdir(tile_images):
                    img_path = f"{tile_images}/{path}"
                    future = executor.submit(self.process_single_image, img_path)
                    futures.append(future)
            for future in tqdm(futures, desc="识别手牌", unit="张"):
                filename, tile_name = future.result()
                if "error" not in tile_name and tile_name != "back":
                    temp_tiles.append(tile_name)
            valid_tiles[key] = temp_tiles
            temp_tiles = []
        return valid_tiles

    def get_dora_indicator_path(self) -> str:
        """获取最新的宝牌指示牌路径"""
        dora_files = sorted(
            Path(profile['PATH']['Split_FinalPath'] + "/" + self.folder_list['Dora_Indicator']).glob("*"),
            key=lambda x: x.stat().st_mtime,  # 按修改时间排序
            reverse=True  # 取最新文件
        )
        return str(dora_files[0]) if dora_files else ""

    def calculate_real_dora(self, indicator_tile: str) -> str:
        """计算真正的宝牌（考虑风牌和三元牌顺序）"""
        if not indicator_tile or indicator_tile == "back":
            return "unknown"
        
        # 分离数字和类型
        num_str = indicator_tile[:-1]
        tile_type = indicator_tile[-1]
        
        try:
            if tile_type in ["m", "p", "s"]:  # 数牌
                num = int(num_str)
                real_num = (num % 9) + 1
                return f"{real_num}{tile_type}"
            elif tile_type == "z":  # 字牌
                z_num = int(num_str)
                # 风牌循环顺序：东(1z)->南(2z)->西(3z)->北(4z)->东
                wind_order = {1: 2, 2: 3, 3: 4, 4: 1}
                # 三元牌循环顺序：白(5z)->发(6z)->中(7z)->白
                dragon_order = {5: 6, 6: 7, 7: 5}
                
                if 1 <= z_num <= 4:  # 风牌
                    return f"{wind_order[z_num]}z"
                elif 5 <= z_num <= 7:  # 三元牌
                    return f"{dragon_order[z_num]}z"
                else:
                    return "unknown"
        except (ValueError, KeyError):
            pass
        return "unknown"

    def recognize_dora(self) -> List[str]:
        """识别宝牌指示牌并计算真实宝牌"""
        dora_path = self.get_dora_indicator_path()
        if not dora_path:
            return []
        
        try:
            # 识别指示牌
            img = cv2.imread(dora_path)
            if img is None:
                return []
            
            indicator_tile = self.classifier(img)
            real_dora = self.calculate_real_dora(indicator_tile)
            return [real_dora] if real_dora != "unknown" else []
        except Exception as e:
            print(f"宝牌识别失败: {str(e)}")
            return []

    def generate_game_state(self) -> Dict:
        """生成游戏状态JSON结构"""
        return {
            "id": -1,
            "state": "GameStart",
            "seatList": [1, 2, 3, 17457800],  # 需根据实际游戏数据修改
            "tiles": self.process_tiles(),
            "doras": self.recognize_dora()  # 使用真实宝牌
        }
    
    # 删除SceenShotPath、Split_FinalPath、Split_FirstPath下所有文件
    def delete_folders(self):
        # Path_list = ["ScreenShotPath", "Split_FinalPath", "Split_FirstPath"]
        Path_list = ["Split_FinalPath", "Split_FirstPath"]
        for folder in Path_list:
            target_dir = profile['PATH'][folder]
            
            # 确保路径存在
            if not os.path.exists(target_dir):
                continue
                
            # 遍历目录内容
            for entry in os.listdir(target_dir):
                full_path = os.path.join(target_dir, entry)
                
                try:
                    if os.path.isfile(full_path) or os.path.islink(full_path):
                        os.remove(full_path)  # 删除文件或符号链接
                    elif os.path.isdir(full_path):
                        shutil.rmtree(full_path)  # 递归删除目录
                except Exception as e:
                    print(f"删除 {full_path} 失败，错误：{e}")


    def save_game_state(self, output_path: str):
        """保存游戏状态到JSON文件"""
        game_state = self.generate_game_state()
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(game_state, f, indent=2, ensure_ascii=False)
            
        print(f"游戏状态已保存至：{os.path.abspath(output_path)}")
        self.delete_folders()
        print("图片删除完成")