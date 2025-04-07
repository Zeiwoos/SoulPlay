from collections import Counter
import re
import os
import json
import time
from typing import Dict, List, Optional, Tuple, Set

def list_subtract(curr_list: List[str], prev_list: List[str]) -> List[str]:
    """计算 curr_list 相对于 prev_list 新增的元素（考虑重复元素）"""
    prev_counter = Counter(prev_list)
    curr_counter = Counter(curr_list)
    new_tiles = []
    for tile, count in curr_counter.items():
        prev_count = prev_counter.get(tile, 0)
        if count > prev_count:
            new_tiles.extend([tile] * (count - prev_count))
    return new_tiles

class MahjongActionDetector:
    def __init__(self):
        self.prev_state = None
        self.tile_pattern = re.compile(r"^(?:([0-9])([mps])|([1-7])(z))$")
        self.wind_map = {"1z": "东", "2z": "南", "3z": "西", "4z": "北"}
        self.last_discard_seat = None
        self.last_discard_tile = None
        self.melds = {pos: [] for pos in ["Self_Mingpai", "Second_Mingpai", "Third_Mingpai", "Fourth_Mingpai"]}
        self.discards = {pos: [] for pos in ["Self_Discard", "Second_Discard", "Third_Discard", "Fourth_Discard"]}
        self.hand_tiles = []
        self.field_wind = None
        self.self_wind = None
        self.action_history = []
        # 存储每个玩家的明牌集合，用于更好地追踪历史
        self.player_melds = {0: [], 1: [], 2: [], 3: []}
        self.current_turn = 0
        self.last_actions = []  # 存储所有历史动作
        self.seat_list = []
        self.turn_order = [] # 座位顺序, 0-3,自己为0
        self.waiting_for_discard = False
        self.next_expected_turn = None
        self.last_meld_check = {}  # 用于跟踪上次检查的明牌状态
    
    def clearAll(self):
        """清理状态"""
        self.prev_state = None
        self.last_discard_seat = None
        self.last_discard_tile = None
        self.hand_tiles = []
        self.field_wind = None
        self.self_wind = None
        self.action_history = []
        # 存储每个玩家的明牌集合，用于更好地追踪历史
        self.player_melds = {0: [], 1: [], 2: [], 3: []}
        self.current_turn = 0
        self.last_actions = []  # 存储所有历史动作
        self.seat_list = []
        self.turn_order = [] # 座位顺序, 0-3,自己为0
        self.waiting_for_discard = False
        self.next_expected_turn = None
        self.last_meld_check = {}  # 用于跟踪上次检查的明牌状态

    def parse_tile(self, tile: str) -> Dict:
        """解析牌的类型和数字"""
        match = self.tile_pattern.match(tile)
        if not match:
            raise ValueError(f"Invalid tile: {tile}")
        if match.group(3):
            return {"num": int(match.group(3)), "type": "z"}
        else:
            return {"num": int(match.group(1)), "type": match.group(2)}
        
    def get_seat_by_position(self, position: str) -> int:
        """根据位置获取座位号"""
        return {"Self_Mingpai":0, "Second_Mingpai":1, "Third_Mingpai":2, "Fourth_Mingpai":3}.get(position, -1)

    def get_position_by_seat(self, seat: int) -> str:
        """根据座位号获取位置"""
        return ["Self_Mingpai", "Second_Mingpai", "Third_Mingpai", "Fourth_Mingpai"][seat] if 0<=seat<=3 else ""

    def get_discard_position_by_seat(self, seat: int) -> str:
        """根据座位号获取弃牌位置"""
        return ["Self_Discard", "Second_Discard", "Third_Discard", "Fourth_Discard"][seat] if 0<=seat<=3 else ""
    
    def Is_states_equal(self, state1: Dict, state2: Dict) -> bool:
        if not state1 or not state2:
            return False
        for key in set(state1.get("tiles", {}).keys()) | set(state2.get("tiles", {}).keys()):
            if Counter(state1.get("tiles", {}).get(key, [])) != Counter(state2.get("tiles", {}).get(key, [])):
                return False
        return True
    
    def detect_seat_order(self, curr_state: Dict):
        """检测座位顺序"""
        seat_list = curr_state.get("seatList", [])
        if not seat_list or len(seat_list) != 4:
            return
        self.seat_list = seat_list.copy()
        if "SelfWind" in curr_state:
            self_wind = curr_state["SelfWind"]
            self_index = int(self_wind[0]) - 1 if self_wind in ["1z", "2z", "3z", "4z"] else 0
            self.turn_order = [(i - self_index) % 4 for i in range(4)]
            print(self.turn_order)
            self.current_turn = self.turn_order[0]
        else:
            self.turn_order = [0, 1, 2, 3]
            self.current_turn = 0

    def find_new_melds(self, prev_melds: List[List[str]], curr_melds: List[List[str]]) -> List[List[str]]:
        """找出新增的面子"""
        new_melds = []
        for curr_meld in curr_melds:
            if not any(Counter(prev_meld)==Counter(curr_meld) for prev_meld in prev_melds):
                new_melds.append(curr_meld)
        return new_melds
    
    def detect_added_kan(self, prev_tiles: List[str], curr_tiles: List[str]) -> Optional[List[str]]:
        """检测是否是在已有三张牌上加杠"""
        # 检查新增了哪些牌
        new_tiles = list_subtract(curr_tiles, prev_tiles)
        print(f"检查是否有加杠新增牌: {new_tiles}")
        if len(new_tiles) != 1:
            return None
            
        # 检查新增的牌是否能与之前的牌组成4张相同的牌
        added_tile = new_tiles[0]
        count = Counter(curr_tiles)[added_tile]
        if count == 4:
            return [added_tile] * 4
            
        # 之前可能已经有3个相同的牌
        prev_triplets = self.identify_triplets(prev_tiles)
        for triplet in prev_triplets:
            if triplet[0] == added_tile:
                return [added_tile] * 4
            
        return None
    
    def is_concealed_kan(self, new_tiles: List[str], visible_tiles: List[str]) -> bool:
        """检测是否是暗杠（可能只能看到两张牌）"""
        # 暗杠通常只显示两张牌
        if len(new_tiles) != 2:
            return False
            
        # 检查这两张牌是否相同
        if new_tiles[0] != new_tiles[1]:
            return False
            
        # 查看之前是否已经有这种牌，如果有可能是加杠而不是暗杠
        for tile in visible_tiles:
            if tile == new_tiles[0]:
                return False
                
        return True
        
    def identify_triplets(self, tiles: List[str]) -> List[List[str]]:
        """只识别刻子（三张相同的牌）"""
        if not tiles:
            return []
            
        triplets = []
        counter = Counter(tiles)
        for tile, count in counter.items():
            if count >= 3:
                triplets.append([tile] * 3)
        return triplets

    def identify_straights(self, tiles: List[str], exclude_tiles: Set[str]) -> List[List[str]]:
        """识别顺子，排除已经用于刻子的牌"""
        if not tiles:
            return []
            
        straights = []
        # 按花色分组
        by_type = {}
        for tile in tiles:
            if tile in exclude_tiles:
                continue
            t = self.parse_tile(tile)
            if t["type"] in ['m', 'p', 's']:  # 只有万、筒、条可以组成顺子
                key = t["type"]
                by_type.setdefault(key, []).append(tile)
        
        # 检查每种花色中可能的顺子
        for suit, suit_tiles in by_type.items():
            # 统计每个数字的牌的数量
            num_counts = {}
            for tile in suit_tiles:
                num = int(tile[0])
                num_counts[num] = num_counts.get(num, 0) + 1
            
            # 寻找连续的三个数字
            for i in range(1, 8):  # 数字1-7才可能是顺子的开始
                if num_counts.get(i, 0) > 0 and num_counts.get(i+1, 0) > 0 and num_counts.get(i+2, 0) > 0:
                    straight = [f"{i}{suit}", f"{i+1}{suit}", f"{i+2}{suit}"]
                    straights.append(straight)
                    # 减少计数，避免同一张牌被多次使用
                    num_counts[i] -= 1
                    num_counts[i+1] -= 1
                    num_counts[i+2] -= 1
        
        return straights

    def identify_melds(self, tiles: List[str]) -> List[List[str]]:
        """更准确地识别面子，优先考虑刻子和杠"""
        if not tiles:
            return []
        
        melds = []
        # 先识别所有可能的刻子
        counter = Counter(tiles)
        kans = []  # 杠
        triplets = []  # 刻子
        
        # 先检查杠（4张相同的牌）
        for tile, count in counter.items():
            if count >= 4:
                kans.append([tile] * 4)
                counter[tile] -= 4  # 减少计数
        
        # 然后检查刻子（3张相同的牌）
        for tile, count in counter.items():
            if count >= 3:
                triplets.append([tile] * 3)
                counter[tile] -= 3  # 减少计数
        
        # 添加杠和刻子到结果
        melds.extend(kans)
        melds.extend(triplets)
        
        # 计算已使用的牌
        used_tiles = set()
        for meld in melds:
            for i, tile in enumerate(meld):
                # 只标记实际数量的牌为已使用
                if counter[tile] > 0:
                    used_tiles.add(tile)
                    counter[tile] -= 1
        
        # 最后检查顺子，排除已用于刻子和杠的牌
        remaining_tiles = [t for t in tiles if t not in used_tiles or counter[t] > 0]
        straights = self.identify_straights(remaining_tiles, used_tiles)
        melds.extend(straights)
        
        return melds

    def detect_melds(self, prev_state: Dict, curr_state: Dict) -> List[Dict]:
        """检测明牌和杠"""
        actions = []
        
        for position in self.melds.keys():
            prev_tiles = prev_state.get("tiles", {}).get(position, [])
            curr_tiles = curr_state.get("tiles", {}).get(position, [])
            
            # 如果没有变化，跳过
            if prev_tiles == curr_tiles:
                continue
                
            # 检查新增的牌
            new_tiles = list_subtract(curr_tiles, prev_tiles)
            print(f"新：{curr_tiles}，\n旧：{prev_tiles}")
            print(f"{position}新增牌: {new_tiles}")
            if not new_tiles:
                print("没有新增牌，跳过")
                continue
            
            seat = self.get_seat_by_position(position)
            
            # 检查是否存在加杠操作（在已有刻子上加一张牌）
            added_kan = self.detect_added_kan(prev_tiles, curr_tiles)
            if added_kan:
                action = {
                    "state": "MyAction_Chipongang" if seat == 0 else "Other_Chipongang",
                    "seat": seat,
                    "tile": new_tiles[0],
                    "doras": curr_state.get("doras", []),
                    "operation": {
                        "type": 5,  # 加杠
                        "combination": added_kan
                    }
                }
                actions.append(action)
                self.current_turn = seat
                self.next_expected_turn = (seat + 1) % 4
                continue
                
            # 检查是否存在暗杠操作（只看到两张相同的牌）
            if self.is_concealed_kan(new_tiles, prev_tiles):
                action = {
                    "state": "MyAction_Chipongang" if seat == 0 else "Other_Chipongang",
                    "seat": seat,
                    "tile": new_tiles[0],
                    "doras": curr_state.get("doras", []),
                    "operation": {
                        "type": 4,  # 暗杠
                        "combination": [new_tiles[0]] * 4
                    }
                }
                actions.append(action)
                self.current_turn = seat
                self.next_expected_turn = (seat + 1) % 4
                continue
            
            # 其他面子检测（刻子、顺子）
            if len(new_tiles) >= 3:  # 至少需要3张牌才能形成面子
                # 使用改进后的面子识别方法
                prev_melds = self.identify_melds(prev_tiles)
                curr_melds = self.identify_melds(curr_tiles)
                new_melds = self.find_new_melds(prev_melds, curr_melds)
                
                for meld in new_melds:
                    action_type = None
                    if len(meld) == 3:
                        if len(set(meld)) == 1:
                            action_type = 3  # 碰
                        else:
                            action_type = 2  # 吃
                    elif len(meld) == 4:
                        action_type = 5  # 明杠
                    if action_type:
                        action = {
                            "state": "MyAction_Chipongang" if seat == 0 else "Other_Chipongang",
                            "seat": seat,
                            "tile": self.last_discard_tile if action_type in [2,3,5] else meld[0],
                            "doras": curr_state.get("doras", []),
                            "operation": {
                                "type": action_type,
                                "combination": meld
                            }
                        }
                        
                        # 避免重复检测，a为当前动作，b为历史动作
                        is_duplicate = any( 
                            Counter(meld) == Counter(a.get("operation", {}).get("combination", []))
                            and a.get("seat") == seat 
                            and a.get("operation", {}).get("type") == action_type
                            for a in self.last_actions
                        )
                        
                        if not is_duplicate:
                            actions.append(action)
                            if action_type in [2,3,5]:
                                self.current_turn = seat
                                self.next_expected_turn = (seat + 1) % 4
                            else:
                                self.next_expected_turn = (self.current_turn + 1) % 4
                        
                # 更新该玩家的面子记录
                self.player_melds[seat] = curr_melds
                
        return actions











    def detect_discards(self, prev_state: Dict, curr_state: Dict) -> List[Dict]:
        actions = []
        current_player_pos = self.get_discard_position_by_seat(self.current_turn)
        if not current_player_pos:
            return actions
        prev_discards = prev_state.get("tiles", {}).get(current_player_pos, [])
        curr_discards = curr_state.get("tiles", {}).get(current_player_pos, [])
        new_discards = list_subtract(curr_discards, prev_discards)
        print(f"弃牌检查新增牌: {new_discards}")
        for tile in new_discards:
            if self.current_turn == 0:
                prev_hand = prev_state.get("tiles", {}).get("Hand_Tiles", [])
                curr_hand = curr_state.get("tiles", {}).get("Hand_Tiles", [])

                get_tile = tile if tile not in prev_hand else (list_subtract(curr_hand, prev_hand) + [''])[0]
                action = {
                    "state": "MyAction",
                    "tile": tile,
                    "getTile": get_tile,
                }
            else:
                action = {
                    "state": "Discard",
                    "seat": self.current_turn,
                    "tile": tile,
                }
            is_duplicate = any(
                a.get("tile") == tile 
                and a.get("seat", -1) == (self.current_turn if self.current_turn !=0 else -1)
                for a in self.last_actions
            )
            if not is_duplicate:
                actions.append(action)
                self.last_discard_seat = self.current_turn
                self.last_discard_tile = tile
                self.current_turn = (self.current_turn + 1) % 4
                self.next_expected_turn = None
        return actions

    def process(self, curr_state: Dict) -> List[Dict]:
        try:
            if self.Is_states_equal(self.prev_state, curr_state):
                return []
            print(curr_state.get("state"))
            if not self.prev_state or self.prev_state.get("state") == "GameEnd":
                if curr_state.get("state") == "GameStart":
                    print("游戏开始")
                    self.detect_seat_order(curr_state)
                    self.prev_state = curr_state.copy()
                    self.last_actions = []
                    return [{
                        "state": "GameStart",
                        "seatList": curr_state.get("seatList", []),
                        "chang": int(curr_state.get("FieldWind", "1z")[0]),
                        "tiles": curr_state.get("tiles", {}).get("Hand_Tiles", []),
                        "doras": curr_state.get("doras", [])
                    }]
                self.prev_state = curr_state.copy()
                return []
            if curr_state.get("state") == "GameEnd":
                self.clearAll()
                self.prev_state = curr_state.copy()
                return [{"state": "GameEnd"}]
            print("旧状态", self.prev_state)
            print("新状态", curr_state)
            actions = []
            meld_actions = self.detect_melds(self.prev_state, curr_state)
            actions.extend(meld_actions)
            discard_actions = self.detect_discards(self.prev_state, curr_state)
            actions.extend(discard_actions)
            self.prev_state = curr_state.copy()
            self.last_actions.extend(actions.copy())
            return actions
        except Exception as e:
            import traceback
            print(f"Error processing state: {e}")
            print(traceback.format_exc())
            return []

def monitor_json(filename:str, detector, action_file):
    if not os.path.exists(filename):
        print(f"文件 {filename} 不存在。")
        return
    last_mtime = 0
    try:
        last_mtime = os.path.getmtime(filename)
        with open(filename, 'r', encoding='utf-8') as f:
            json_data = f.read().strip()
            prev_data = json.loads(json_data) if json_data else {}
    except Exception as e:
        print(f"初始读取失败: {e}")
        return
    print("开始监视文件变化...")
    while True:
        time.sleep(0.3)
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
            actions = detector.process(curr_data)
            for action in actions:
                with open(action_file, 'a', encoding='utf-8') as f:
                    f.write(f"{json.dumps(action, ensure_ascii=False)}\n")
                action_type = action.get("state")
                seat = action.get("seat", 0)
                tile = action.get("tile", "")
                op_type = action.get("operation", {}).get("type", 0)
                op_name = {1:"弃牌",2:"吃",3:"碰",4:"暗杠",5:"明杠"}.get(op_type, "")
                if action_type == "MyAction":
                    print(f"检测到动作: {action_type} - 自己打出 {tile} 摸到 {action.get('getTile', '')}")
                elif action_type == "Discard":
                    print(f"检测到动作: {action_type} - 玩家{seat}打出 {tile}")
                elif "Chipongang" in action_type:
                    print(f"检测到动作: {action_type} - 玩家{seat}{op_name} {tile}")
        except Exception as e:
            import traceback
            print(f"处理错误: {e}")
            print(traceback.format_exc())

if __name__ == '__main__':
    try:
        with open("Data/json/profile.json", "r", encoding="utf-8") as f:
            profile = json.load(f)
        with open(profile["PATH"]["ActionPath"], "w", encoding="utf-8") as f:
            pass
        detector = MahjongActionDetector()
        monitor_json(profile["PATH"]["BoardStatePath"], detector, profile["PATH"]["ActionPath"])
        # monitor_json(profile["PATH"]["BoardStatePath"], detector, "E:\Programs\Code_VSCode\MahjongCopilot\game_log\pre.txt")
    except Exception as e:
        print(f"启动错误: {e}")
        input("按任意键退出...")