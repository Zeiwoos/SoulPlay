import os
import cv2
import json
from tqdm import tqdm
from pathlib import Path
from typing import List, Dict
from deepdiff import DeepDiff
from concurrent.futures import ThreadPoolExecutor
from IMGProcess.BatchClassify import BatchClassifier
from typing import Optional
from collections import Counter

with open("Data/json/profile.json", "r", encoding="utf-8") as f:
    profile = json.load(f)
class GameStateGenerator(BatchClassifier):
    """
    游戏状态生成器
    """
    def __init__(self, self_wind, field_wind, GameState=None):
        super().__init__()
        self.folder_list = None
        self.parent_folder = profile['PATH']['Split_FinalPath']
        self.last_game_state = {}
        self.SelfWind = self_wind
        self.FieldWind = field_wind
        self.GameState = GameState
        self.seatlist = [1, 2, 3, 17457800]
        self.seat_map = {}
        self.reverse_seat_map = []

    def find_subfolders_with_suffix_scandir(self, filename: str) -> None:
        """使用 os.scandir() 高效查找一级子文件夹是否匹配 filename_后缀"""

        suffixes = profile['Suffix']['Suffix']
        expected_names = {f"{filename}_{sfx}": sfx for sfx in suffixes}
        folder_list = {sfx: None for sfx in suffixes}

        try:
            for entry in os.scandir(self.parent_folder):
                if entry.is_dir() and entry.name in expected_names:
                    suffix = expected_names[entry.name]
                    folder_list[suffix] = entry.name
        except FileNotFoundError:
            print(f"❌ 文件夹路径不存在: {self.parent_folder}")
        except Exception as e:
            print(f"❌ 扫描文件夹时出错: {e}")

        self.folder_list = folder_list

    def update_seat_map(self) -> bool:
        """根据自风和场风更新座位映射关系"""

        # 检查 seatList 是否有效
        if not isinstance(self.seatlist, list) or len(self.seatlist) != 4:
            print("❌ seatList 无效或不包含4个元素")
            return False

        if not self.FieldWind or not self.SelfWind:
            print("❌ 缺少场风或自风信息")
            return False

        # 风字对应的位置（东南西北 → 0~3）
        wind_values = {"1z": 0, "2z": 1, "3z": 2, "4z": 3}

        try:
            field_value = wind_values[self.FieldWind]
            self_value = wind_values[self.SelfWind]
        except KeyError:
            print(f"❌ 无法识别风位: Field={self.FieldWind}, Self={self.SelfWind}")
            return False

        try:
            # 当前玩家在 seatlist 中的位置
            my_index = self.seatlist.index(max(self.seatlist))  # 最大值视为自己
            my_pos = self_value  # 自己在东南西北中所处的逻辑位置

            # 生成位置映射（逻辑位置 → seatlist 下标）
            seat_mapping = {my_pos: my_index}
            other_indexes = [i for i in range(4) if i != my_index]

            # 顺时针安排其他玩家的位置
            for offset, idx in enumerate(other_indexes, start=1):
                seat_mapping[(my_pos + offset) % 4] = idx

            self.seat_map = seat_mapping

            # 构建按东南西北顺序排列的 seatList
            self.reverse_seat_map = [self.seatlist[seat_mapping[i]] for i in range(4)]

            return True

        except Exception as e:
            print(f"❌ 更新座位映射失败: {e}")
            return False


    def process_tiles(self) -> Dict[str, List[str]]:
        """多线程处理各类麻将图片，返回每类牌的识别结果"""
        print("🀄 正在识别手牌...")
        valid_tiles = {}

        for key, folder in self.folder_list.items():
            if key in ("Dora_Indicator", "Wind"):
                continue
            
            # 文件夹不存在时跳过该类牌
            if not folder:
                valid_tiles[key] = []
                continue

            tile_folder_path = Path(profile['PATH']['Split_FinalPath']) / folder
            if not tile_folder_path.exists() or not tile_folder_path.is_dir():
                print(f"⚠️ 牌面文件夹不存在或无效: {tile_folder_path}")
                valid_tiles[key] = []
                continue

            tile_paths = list(tile_folder_path.iterdir())
            if not tile_paths:
                valid_tiles[key] = []
                continue

            # 多线程识别
            tile_results = []
            with ThreadPoolExecutor(max_workers=4) as executor:
                futures = {executor.submit(self.process_single_image, str(img_path)): img_path.name for img_path in tile_paths}

                for future in tqdm(futures, desc=f"识别 {key}", unit="张"):
                    try:
                        filename, tile_name = future.result()
                        if tile_name not in ("back", "error") and "error" not in tile_name:
                            tile_results.append(tile_name)
                    except Exception as e:
                        print(f"❌ 识别失败：{futures[future]}，错误信息：{e}")

            valid_tiles[key] = tile_results

        return valid_tiles


    def get_dora_indicator_path(self) -> str:
        """获取最新的宝牌指示牌图片路径"""
        folder_name = self.folder_list.get("Dora_Indicator")
        if not folder_name:
            print("⚠️ 未找到宝牌指示牌文件夹（Dora_Indicator）")
            return None
        
        dora_path = Path(profile['PATH']['Split_FinalPath']) / folder_name
        if not dora_path.exists() or not dora_path.is_dir():
            print(f"⚠️ 路径不存在或不是文件夹: {dora_path}")
            return None
        
        dora_files = sorted(
            dora_path.glob("*"),
            key=lambda x: x.stat().st_mtime,
            reverse=True
        )

        if not dora_files:
            print(f"⚠️ 宝牌指示牌文件夹中无图片文件: {dora_path}")
            return ""

        return str(dora_files[0])


    def calculate_real_dora(self, indicator_tile: str) -> str:
        """
        根据宝牌指示牌计算真正的宝牌。
        - 数牌顺序为：1~9 → 下一张，9 → 1
        - 字牌顺序为：东南西北 → 顺时针；中发白 → 中→发→白→中
        """
        if not indicator_tile or indicator_tile == "back":
            return "unknown"

        if len(indicator_tile) < 2:
            return "unknown"

        num_str = indicator_tile[:-1]
        tile_type = indicator_tile[-1]

        if tile_type not in ["m", "p", "s", "z"]:
            return "unknown"

        try:
            num = int(num_str)
        except ValueError:
            return "unknown"

        if tile_type in ["m", "p", "s"]:  # 数牌
            # 红宝牌“0”视作“5”，对应的正宝为“6”
            return f"{6 if num == 0 else (num % 9) + 1}{tile_type}"
        
        if tile_type == "z":  # 字牌（风牌 + 三元牌）
            # 东(1)→南(2)→西(3)→北(4)→东(1)；中(5)→发(6)→白(7)→中(5)
            dora_order = {1: 2, 2: 3, 3: 4, 4: 1, 5: 6, 6: 7, 7: 5}
            return f"{dora_order.get(num, 'unknown')}{tile_type}"

        return "unknown"


    def recognize_dora(self) -> List[str]:
        """识别宝牌指示牌并计算真实宝牌"""
        print("正在识别宝牌指示牌...")
        dora_path = self.get_dora_indicator_path()
        if not dora_path:
            return None
        try:
            # 识别指示牌
            img = cv2.imread(dora_path)
            if img is None:
                return None
            
            indicator_tile = self.classifier(img)
            real_dora = self.calculate_real_dora(indicator_tile)
            # #无需计算，直接给出即可
            # real_dora = indicator_tile
            return [real_dora] if real_dora != "unknown" else None
        except Exception as e:
            print(f"宝牌识别失败: {str(e)}")
            return None

    # def generate_board_state(self) -> Dict:
    #     """生成游戏状态JSON结构"""
    #     self.update_seat_map()
    #     tiles = self.process_tiles()
    #     doras = self.recognize_dora()
    #     # 手牌或者宝牌都为空时，返回空字典
    #     if tiles is None or doras is None or self.reverse_seat_map is None:
    #         return None
    #     if (len(tiles['Hand_Tiles']) < 13 and self.GameState == "GameStart"):
    #         print("⚠️ 手牌数量不足，无法生成游戏状态")
    #         return None
        
            
    #     return {
    #         "state": self.GameState, # 游戏状态
    #         "FieldWind": self.FieldWind,   # 东南西北
    #         "SelfWind": self.SelfWind, # 自风
    #         "seatList": self.reverse_seat_map,  # 座位顺序始终为东南西北（1Z,2Z,3Z,4Z）
    #         "tiles": tiles, # 麻将牌
    #         "doras": doras # 宝牌
    #         # "tiles": self.process_tiles(), # 麻将牌
    #         # "doras": self.recognize_dora() # 宝牌
    #     }
    def generate_board_state(self) -> Optional[Dict]:
        """生成游戏状态JSON结构"""

        self.update_seat_map()

        # 预处理所需信息
        tiles = self.process_tiles()
        doras = self.recognize_dora()

        # 基本有效性校验
        if tiles is None or doras is None or self.reverse_seat_map is None:
            return None

        hand_tiles = tiles.get('Hand_Tiles', [])
        if self.GameState == "GameStart" and len(hand_tiles) < 13:
            print("⚠️ 手牌数量不足，无法生成游戏状态")
            return None

        # 正常状态，返回结构
        return {
            "state": self.GameState,
            "FieldWind": self.FieldWind,
            "SelfWind": self.SelfWind,
            "seatList": self.reverse_seat_map,
            "tiles": tiles,
            "doras": doras
        }
    def check_tile_counts_valid(self, tiles: Dict[str, List[str]]) -> bool:
        all_tiles = []

        for key, value in tiles.items():
            # 只处理值为 List[str] 的字段
            if isinstance(value, list) and all(isinstance(t, str) for t in value):
                all_tiles.extend(value)

        tile_counter = Counter(all_tiles)

        for tile, count in tile_counter.items():
            if count > 4:
                print(f"⚠️ 牌 {tile} 出现了 {count} 次，超过4张")
                return False
        return True

    def save_board_state(self, output_path: str, verbose: bool = True) -> bool:
        """保存游戏状态到JSON文件"""
        board_state = self.generate_board_state()

        if board_state is None:
            if verbose:
                print("❌ 无法生成当前牌局状态")
            return False
        
        # 牌数量合理性校验
        if not self.check_tile_counts_valid(board_state['tiles']):
            print("⚠️ 检测到某些牌数量超过4张，疑似识别异常,不保存")
            print(f"🀄️ 未保存牌局状态：{board_state}")
            return True

        if verbose:
            print(f"🀄️ 当前牌局状态：{board_state}")

        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(board_state, f, indent=2, ensure_ascii=False)
            if verbose:
                print(f"✅ 牌局状态已保存至：{os.path.abspath(output_path)}")
            return True
        except Exception as e:
            print(f"❌ 保存牌局状态时发生错误: {e}")
            return False