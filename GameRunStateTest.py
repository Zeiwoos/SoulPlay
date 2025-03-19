import cv2
import json
import os

with open("Data/json/profile.json", "r", encoding="utf-8") as f:  # 明确指定 UTF-8 编码
    profile = json.load(f)

best_match_state = profile["BestMatchState"]

def load_templates(template_folder):
    """ 加载指定文件夹内的所有模板 """
    templates = []
    for file in os.listdir(template_folder):
        path = os.path.join(template_folder, file)
        template = cv2.imread(path, 0)  # 读取为灰度图
        if template is not None:
            templates.append(template)
    return templates

def match_game_state(state, screen, templates, threshold=0.6):
    """
    在屏幕截图中匹配一组模板，并返回最佳匹配的状态
    :param screen: 游戏当前屏幕截图（灰度）
    :param templates: 模板图像列表
    :param threshold: 匹配阈值
    :return: 是否匹配上
    """

    best_score = 0
    for template in templates:
        img = cv2.imread(screen, 0)
        result = cv2.matchTemplate(img, template, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
        best_score = max(best_score, max_val)
    # 将最佳匹配度写入profile.json
    with open("Data/json/profile.json", "w", encoding="utf-8") as f:
        profile["BestMatchState"][state] = best_score
        json.dump(profile, f)
    return best_score

def get_game_state(screen):
    # 加载所有模板
    game_states = {}
    for state, folder in profile["Templates"].items():
        game_states[state] = load_templates(folder)

    # 遍历所有游戏状态，检测当前游戏状态
    for state, templates in game_states.items():
        print(f"检测游戏状态: {state}")
        match_game_state(state, screen, templates, threshold=0.6)

    # 获取最佳匹配状态
    game_state = max(best_match_state, key=best_match_state.get)
    print(best_match_state)
    if best_match_state["result_screen"] > 0.9:
        game_state = "result_screen"

    print(f"最佳匹配状态: {game_state}")

    return game_state

