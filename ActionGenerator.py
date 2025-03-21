from collections import Counter
import re
import os
import json
import time
from typing import Dict, List, Optional, Tuple, Set

OUTPUT_FILE = 'Data\\json\\game-state\\Action.txt'

class MahjongActionDetector:
    def __init__(self):
        self.prev_state = None
        self.seat_map = {}
        self.reverse_seat_map = {}
        self.tile_pattern = re.compile(r"^([0-9])([mpsz])$")
        self.wind_map = {"1z": "东", "2z": "南", "3z": "西", "4z": "北"}
        self.last_discard_seat = None
        self.last_discard_tile = None
        self.melds = {
            "Self_Mingpai": set(),
            "Second_Mingpai": set(),
            "Third_Mingpai": set(),
            "Fourth_Mingpai": set()
        }
        self.discards = {
            "Self_Discard": set(),
            "Second_Discard": set(),
            "Third_Discard": set(),
            "Fourth_Discard": set()
        }
        self.hand_tiles = set()
        self.field_wind = None
        self.self_wind = None

    def parse_tile(self, tile: str) -> Dict:
        """解析单张牌面结构"""
        match = self.tile_pattern.match(tile)
        if not match:
            raise ValueError(f"Invalid tile format: {tile}")
        return {"num": int(match.group(1)), "type": match.group(2)}

    def diff_tiles(self, prev: List[str], curr: List[str]) -> Dict:
        """计算牌组差异"""
        if prev is None:
            prev = []
        if curr is None:
            curr = []
        prev_set = set(prev)
        curr_set = set(curr)
        return {
            "added": list(curr_set - prev_set),
            "removed": list(prev_set - curr_set)
        }

    def get_meld_key(self, meld: List[str]) -> str:
        """获取明牌组合的唯一键"""
        return ",".join(sorted(meld))

    def update_melds(self, curr_state: Dict):
        """更新所有明牌组合"""
        # 更新各家的明牌
        for position in ["Self_Mingpai", "Second_Mingpai", "Third_Mingpai", "Fourth_Mingpai"]:
            curr_melds = set()
            for meld in curr_state.get("tiles", {}).get(position, []):
                meld_key = self.get_meld_key(meld)
                curr_melds.add(meld_key)
            self.melds[position] = curr_melds

    def update_discards(self, curr_state: Dict):
        """更新所有弃牌"""
        for position in ["Self_Discard", "Second_Discard", "Third_Discard", "Fourth_Discard"]:
            curr_discards = set(curr_state.get("tiles", {}).get(position, []))
            self.discards[position] = curr_discards

    def update_hand_tiles(self, curr_state: Dict):
        """更新手牌"""
        self.hand_tiles = set(curr_state.get("tiles", {}).get("Hand_Tiles", []))

    def update_winds(self, curr_state: Dict):
        """更新场风和自风"""
        self.field_wind = curr_state.get("FieldWind")
        self.self_wind = curr_state.get("SelfWind")

    def get_seat_from_wind(self, wind: str) -> int:
        """根据风位确定座位号"""
        if not wind or not self.field_wind or not self.self_wind:
            return None
            
        # 东南西北对应的数字
        wind_values = {"1z": 0, "2z": 1, "3z": 2, "4z": 3}
        
        # 计算相对位置
        field_value = wind_values.get(self.field_wind, 0)
        self_value = wind_values.get(self.self_wind, 0)
        wind_value = wind_values.get(wind, 0)
        
        # 计算座位号
        relative_pos = (wind_value - field_value) % 4
        my_pos = (self_value - field_value) % 4
        
        # 计算相对于自己的位置
        seat = (relative_pos - my_pos) % 4
        return seat

    def get_position_from_seat(self, seat: int) -> str:
        """根据座位号获取对应的位置标识"""
        position_map = {
            0: "Self",
            1: "Second",
            2: "Third",
            3: "Fourth"
        }
        return position_map.get(seat, "Unknown")

    def detect_new_meld(self, prev_state: Dict, curr_state: Dict) -> Optional[Dict]:
        """检测新的明牌组合"""
        if not prev_state or not curr_state:
            return None

        prev_melds = {}
        curr_melds = {}
        
        # 获取之前和当前的所有明牌组合
        for position in ["Self_Mingpai", "Second_Mingpai", "Third_Mingpai", "Fourth_Mingpai"]:
            prev_melds[position] = {self.get_meld_key(meld) for meld in prev_state.get("tiles", {}).get(position, [])}
            curr_melds[position] = {self.get_meld_key(meld) for meld in curr_state.get("tiles", {}).get(position, [])}
        
        # 检查自己的明牌变化
        new_self_melds = curr_melds["Self_Mingpai"] - prev_melds["Self_Mingpai"]
        if new_self_melds:
            # 找到新增的明牌组合
            for meld_key in new_self_melds:
                meld = meld_key.split(",")
                
                # 暗杠的特殊处理
                if len(meld) == 4 and len(set(meld)) == 1 and (self.last_discard_tile is None or self.last_discard_tile not in meld):
                    # 暗杠
                    return {
                        "state": "MyAction_Chipongang",
                        "tile": meld[0],
                        "doras": curr_state.get("doras", []),
                        "operation": {
                            "type": 4,  # 暗杠
                            "combination": meld,
                            "form": [0, 0, 0]
                        }
                    }
                
                # 如果是最后一张弃牌，很可能是吃/碰/杠
                if self.last_discard_tile and self.last_discard_tile in meld:
                    # 确定组合类型
                    if len(meld) == 3:
                        # 检查是顺子(吃)还是刻子(碰)
                        tiles = [self.parse_tile(t) for t in meld]
                        if len(set(meld)) == 1 or (len(set(meld)) == 3 and all(t["num"] == tiles[0]["num"] for t in tiles)):
                            # 碰
                            form = [0, 0, 0]
                            for i in range(len(meld)):
                                if meld[i] == self.last_discard_tile:
                                    form[i] = self.last_discard_seat
                            return {
                                "state": "MyAction_Chipongang",
                                "tile": self.last_discard_tile,
                                "doras": curr_state.get("doras", []),
                                "operation": {
                                    "type": 3,  # 碰
                                    "combination": meld,
                                    "form": form
                                }
                            }
                        elif all(t["type"] == tiles[0]["type"] for t in tiles) and tiles[0]["type"] != "z":
                            # 检查是否为顺子(吃)
                            nums = sorted(t["num"] for t in tiles)
                            if nums[1] - nums[0] == 1 and nums[2] - nums[1] == 1:
                                # 吃
                                form = [0, 0, 0]
                                for i in range(len(meld)):
                                    if meld[i] == self.last_discard_tile:
                                        form[i] = self.last_discard_seat
                                return {
                                    "state": "MyAction_Chipongang",
                                    "tile": self.last_discard_tile,
                                    "doras": curr_state.get("doras", []),
                                    "operation": {
                                        "type": 2,  # 吃
                                        "combination": meld,
                                        "form": form
                                    }
                                }
                    elif len(meld) == 4:
                        # 杠
                        if len(set(meld)) == 1:
                            # 明杠
                            form = [0, 0, 0, 0]
                            for i in range(len(meld)):
                                if meld[i] == self.last_discard_tile:
                                    form[i] = self.last_discard_seat
                            return {
                                "state": "MyAction_Chipongang",
                                "tile": self.last_discard_tile,
                                "doras": curr_state.get("doras", []),
                                "operation": {
                                    "type": 5,  # 大明杠
                                    "combination": meld,
                                    "form": form[:3]  # 只需要前三个
                                }
                            }

        # 检查其他玩家的明牌变化
        for position in ["Second_Mingpai", "Third_Mingpai", "Fourth_Mingpai"]:
            seat = int(position[0]) - 1 if position[0].isdigit() else 0
            if position == "Second_Mingpai":
                seat = 1  # 下家
            elif position == "Third_Mingpai":
                seat = 2  # 对家
            elif position == "Fourth_Mingpai":
                seat = 3  # 上家
            
            new_melds = curr_melds[position] - prev_melds[position]
            
            if new_melds:
                for meld_key in new_melds:
                    meld = meld_key.split(",")
                    
                    # 暗杠的特殊处理
                    if len(meld) == 4 and len(set(meld)) == 1 and (self.last_discard_tile is None or self.last_discard_tile not in meld):
                        # 暗杠
                        return {
                            "state": "Other_Chipongang",
                            "seat": seat,
                            "tile": meld[0],
                            "doras": curr_state.get("doras", []),
                            "operation": {
                                "type": 4,  # 暗杠
                                "combination": meld,
                                "form": [0, 0, 0]
                            }
                        }
                    
                    # 如果是最后一张弃牌，很可能是吃/碰/杠
                    if self.last_discard_tile and self.last_discard_tile in meld:
                        # 确定组合类型
                        if len(meld) == 3:
                            # 检查是顺子(吃)还是刻子(碰)
                            tiles = [self.parse_tile(t) for t in meld]
                            if len(set(meld)) == 1 or (len(set(meld)) == 3 and all(t["num"] == tiles[0]["num"] for t in tiles)):
                                # 碰
                                form = [0, 0, 0]
                                for i in range(len(meld)):
                                    if meld[i] == self.last_discard_tile:
                                        form[i] = self.last_discard_seat
                                return {
                                    "state": "Other_Chipongang",
                                    "seat": seat,
                                    "tile": self.last_discard_tile,
                                    "doras": curr_state.get("doras", []),
                                    "operation": {
                                        "type": 3,  # 碰
                                        "combination": meld,
                                        "form": form
                                    }
                                }
                            elif all(t["type"] == tiles[0]["type"] for t in tiles) and tiles[0]["type"] != "z":
                                # 检查是否为顺子(吃)
                                nums = sorted(t["num"] for t in tiles)
                                if nums[1] - nums[0] == 1 and nums[2] - nums[1] == 1:
                                    # 吃 (只有下家可以吃)
                                    if seat == 1:  # 下家
                                        form = [0, 0, 0]
                                        for i in range(len(meld)):
                                            if meld[i] == self.last_discard_tile:
                                                form[i] = self.last_discard_seat
                                        return {
                                            "state": "Other_Chipongang",
                                            "seat": seat,
                                            "tile": self.last_discard_tile,
                                            "doras": curr_state.get("doras", []),
                                            "operation": {
                                                "type": 2,  # 吃
                                                "combination": meld,
                                                "form": form
                                            }
                                        }
                        elif len(meld) == 4:
                            # 杠
                            if len(set(meld)) == 1:
                                # 明杠
                                form = [0, 0, 0, 0]
                                for i in range(len(meld)):
                                    if meld[i] == self.last_discard_tile:
                                        form[i] = self.last_discard_seat
                                return {
                                    "state": "Other_Chipongang",
                                    "seat": seat,
                                    "tile": self.last_discard_tile,
                                    "doras": curr_state.get("doras", []),
                                    "operation": {
                                        "type": 5,  # 大明杠
                                        "combination": meld,
                                        "form": form[:3]  # 只需要前三个
                                    }
                                }
        
        return None

    def detect_self_action(self, prev_state: Dict, curr_state: Dict) -> Optional[Dict]:
        """检测自己的出牌动作"""
        if not prev_state or not curr_state:
            return None
            
        prev_hand = set(prev_state.get("tiles", {}).get("Hand_Tiles", []))
        curr_hand = set(curr_state.get("tiles", {}).get("Hand_Tiles", []))
        
        prev_discard = set(prev_state.get("tiles", {}).get("Self_Discard", []))
        curr_discard = set(curr_state.get("tiles", {}).get("Self_Discard", []))
        
        # 检查是否有新弃牌
        new_discards = curr_discard - prev_discard
        if new_discards:
            # 自己打出了牌
            discard_tile = list(new_discards)[0]
            self.last_discard_tile = discard_tile
            self.last_discard_seat = 0  # 自己
            print(f"{curr_hand}\n{prev_hand}")

            # 检查是否有新摸牌
            new_tiles = curr_hand - prev_hand
            removed_tiles = prev_hand - curr_hand
            
            # 如果有新摸牌，记录下来
            get_tile = list(new_tiles)[0] if new_tiles else discard_tile
            
            return {
                "state": "MyAction",
                "tile": discard_tile,
                "getTile": get_tile,
                "doras": curr_state.get("doras", [])
            }
        
        return None

    def detect_other_discard(self, prev_state: Dict, curr_state: Dict) -> Optional[Dict]:
        """检测其他玩家的弃牌"""
        if not prev_state or not curr_state:
            return None
            
        # 检查各个玩家的弃牌堆
        positions = ["Second_Discard", "Third_Discard", "Fourth_Discard"]
        seat_map = {
            "Second_Discard": 1,  # 下家
            "Third_Discard": 2,   # 对家
            "Fourth_Discard": 3   # 上家
        }
        
        for position in positions:
            seat = seat_map[position]
            
            prev_discard = set(prev_state.get("tiles", {}).get(position, []))
            curr_discard = set(curr_state.get("tiles", {}).get(position, []))
            
            # 检查是否有新弃牌
            new_discards = curr_discard - prev_discard
            if new_discards:
                # 有新弃牌
                discard_tile = list(new_discards)[0]
                self.last_discard_tile = discard_tile
                self.last_discard_seat = seat
                
                return {
                    "state": "Discard",
                    "seat": seat,
                    "tile": discard_tile,
                    "doras": curr_state.get("doras", []),
                    "operation": {"type": 1}
                }
        
        return None

    def detect_added_kan(self, prev_state: Dict, curr_state: Dict) -> Optional[Dict]:
        """检测加杠"""
        if not prev_state or not curr_state:
            return None
            
        # 检查所有玩家的明牌
        positions = ["Self_Mingpai", "Second_Mingpai", "Third_Mingpai", "Fourth_Mingpai"]
        seat_map = {
            "Self_Mingpai": 0,     # 自己
            "Second_Mingpai": 1,   # 下家
            "Third_Mingpai": 2,    # 对家
            "Fourth_Mingpai": 3    # 上家
        }
        
        for position in positions:
            seat = seat_map[position]
            
            prev_melds = []
            curr_melds = []
            
            # 获取之前和当前的所有明牌组合
            for meld in prev_state.get("tiles", {}).get(position, []):
                prev_melds.append(sorted(meld))
                
            for meld in curr_state.get("tiles", {}).get(position, []):
                curr_melds.append(sorted(meld))
            
            # 检查是否有三张变四张的情况（加杠）
            for prev_meld in prev_melds:
                if len(prev_meld) == 3 and len(set(prev_meld)) == 1:
                    # 找到三张相同的牌
                    for curr_meld in curr_melds:
                        if len(curr_meld) == 4 and len(set(curr_meld)) == 1 and curr_meld[0] == prev_meld[0]:
                            # 找到四张相同的牌，且与之前的三张匹配
                            if seat == 0:  # 自己
                                return {
                                    "state": "MyAction_Chipongang",
                                    "tile": curr_meld[0],
                                    "doras": curr_state.get("doras", []),
                                    "operation": {
                                        "type": 6,  # 加杠
                                        "combination": curr_meld,
                                        "form": [0, 0, 0]
                                    }
                                }
                            else:
                                return {
                                    "state": "Other_Chipongang",
                                    "seat": seat,
                                    "tile": curr_meld[0],
                                    "doras": curr_state.get("doras", []),
                                    "operation": {
                                        "type": 6,  # 加杠
                                        "combination": curr_meld,
                                        "form": [0, 0, 0]
                                    }
                                }
        
        return None

    def update_seat_map(self, curr_state: Dict):
        """更新座位映射"""
        try:
            # 更新场风和自风
            self.update_winds(curr_state)
            
            if "seatList" in curr_state:
                seatList = curr_state["seatList"]
                if len(seatList) == 4:
                    # 根据场风和自风确定座位映射
                    # 东南西北对应的数字
                    wind_values = {"1z": 0, "2z": 1, "3z": 2, "4z": 3}
                    
                    # 获取场风和自风
                    field_wind = curr_state.get("FieldWind")
                    self_wind = curr_state.get("SelfWind")
                    
                    if field_wind and self_wind:
                        field_value = wind_values.get(field_wind, 0)
                        self_value = wind_values.get(self_wind, 0)
                        
                        # 计算自己的实际座位
                        my_pos = (self_value - field_value) % 4
                        
                        # 确定其他座位
                        self.seat_map = {
                            0: my_pos,                  # 自己
                            1: (my_pos + 1) % 4,        # 下家
                            2: (my_pos + 2) % 4,        # 对家
                            3: (my_pos + 3) % 4         # 上家
                        }
                        
                        # 存储实际座位ID
                        self.reverse_seat_map = {v: seatList[v] for v in range(4)}
                    else:
                        # 如果没有风位信息，使用默认映射
                        self.seat_map = {
                            0: 0,  # 自己
                            1: 1,  # 下家
                            2: 2,  # 对家
                            3: 3   # 上家
                        }
                        
                        # 存储实际座位ID
                        self.reverse_seat_map = {
                            0: seatList[0],  # 自己
                            1: seatList[1],  # 下家
                            2: seatList[2],  # 对家
                            3: seatList[3]   # 上家
                        }
        except Exception as e:
            print(f"更新座位映射错误: {e}")

    def process(self, curr_state: Dict) -> Optional[Dict]:
        """处理当前状态，返回检测到的动作"""
        try:
            # 初始状态处理
            if not self.prev_state:
                if "state" in curr_state and curr_state["state"] == "GameStart":
                    self.update_seat_map(curr_state)
                    return {
                        "state": "GameStart",
                        "seatList": curr_state.get("seatList", []),
                        "chang": int(curr_state.get("FieldWind", "1z")[0]),
                        "tiles": curr_state.get("tiles", {}).get("Hand_Tiles", []),
                        "doras": curr_state.get("doras", [])
                    }
                self.prev_state = curr_state
                return None

            # 游戏结束检测
            if "state" in curr_state and curr_state["state"] == "GameEnd":
                return {"state": "GameEnd"}

            # 更新座位映射
            self.update_seat_map(curr_state)
            
            # 更新牌堆信息
            self.update_melds(curr_state)
            self.update_discards(curr_state)
            self.update_hand_tiles(curr_state)

            # 按优先级检测动作
            detected_action = (
                self.detect_new_meld(self.prev_state, curr_state) or     # 吃/碰/杠
                self.detect_added_kan(self.prev_state, curr_state) or    # 加杠
                self.detect_self_action(self.prev_state, curr_state) or  # 自己出牌
                self.detect_other_discard(self.prev_state, curr_state)   # 其他人出牌
            )

            self.prev_state = curr_state
            return detected_action or {"state": curr_state.get("state", "GameRunning")}
        except Exception as e:
            print(f"处理状态错误: {e}")
            return None

def monitor_json(filename, detector):
    """监控JSON文件变化，并处理检测到的动作"""
    if not os.path.exists(filename):
        print(f"文件 {filename} 不存在。")
        return

    last_mtime = 0
    try:
        last_mtime = os.path.getmtime(filename)
        with open(filename, 'r', encoding='utf-8') as f:
            json_data = f.read().strip()
            if json_data:
                prev_data = json.loads(json_data)
            else:
                prev_data = {}
    except Exception as e:
        print(f"初始读取失败: {e}")
        return

    print("开始监视文件变化...")
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        f.write("# 麻将动作监测日志\n")  # 清空并初始化输出文件
        
    while True:
        time.sleep(0.5)  # 减少检查间隔，提高响应速度
        try:
            current_mtime = os.path.getmtime(filename)
            if current_mtime == last_mtime:
                continue

            last_mtime = current_mtime
            with open(filename, 'r', encoding='utf-8') as f:
                json_data = f.read().strip()
                if not json_data:
                    continue
                curr_data = json.loads(json_data)

            action = detector.process(curr_data)
            if action and action["state"] != "GameRunning":  # 不输出 GameRunning 状态
                with open(OUTPUT_FILE, 'a', encoding='utf-8') as f:
                    f.write(f"{json.dumps(action, ensure_ascii=False)}\n")
                print(f"检测到动作: {action['state']}")

        except Exception as e:
            print(f"处理错误: {e}")
            continue

if __name__ == '__main__':
    try:
        with open("Data/json/profile.json", "r", encoding="utf-8") as f:
            profile = json.load(f)
        
        detector = MahjongActionDetector()
        monitor_json(profile["PATH"]["BoardStatePath"], detector)
    except Exception as e:
        print(f"启动错误: {e}")
        input("按任意键退出...")