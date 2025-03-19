import cv2
import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from functools import lru_cache

# 预处理配置
with open("Data/json/profile.json", "r", encoding="utf-8") as f:
    profile = json.load(f)

# 使用LRU缓存避免重复读取模板
@lru_cache(maxsize=32)
def load_templates_cached(template_folder):
    """缓存模板加载结果"""
    templates = []
    for file in sorted(os.listdir(template_folder)):
        path = os.path.join(template_folder, file)
        template = cv2.imread(path, 0)
        if template is not None:
            templates.append((file, template))
    return templates

class GameRunStateDetector:
    def __init__(self):
        # 多线程执行器
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        # 预加载所有模板结构
        self.template_cache = {}
        for state, folder in profile["Templates"].items():
            self.template_cache[state] = load_templates_cached(folder)
        
        # 共享状态锁
        self.lock = threading.Lock()
        self.current_screen = None
        self.best_scores = {}

    def _parallel_match(self, state, screen_gray):
        """并行匹配单个游戏状态"""
        best_score = 0
        for file, template in self.template_cache[state]:
            try:
                result = cv2.matchTemplate(screen_gray, template, cv2.TM_CCOEFF_NORMED)
                _, max_val, _, _ = cv2.minMaxLoc(result)
                if max_val > best_score:
                    best_score = max_val
                    # 提前终止阈值
                    if best_score > 0.95:
                        break
            except cv2.error:
                continue
        
        with self.lock:
            self.best_scores[state] = best_score

    def _update_profile(self):
        """批量更新配置文件"""
        with self.lock:
            profile["BestMatchState"].update(self.best_scores)
            with open("Data/json/profile.json", "w", encoding="utf-8") as f:
                json.dump(profile, f, ensure_ascii=False, indent=2)

    def get_game_state(self, screen_path):
        """优化后的游戏状态检测"""
        # 单次读取屏幕截图
        screen_gray = cv2.imread(screen_path, 0)
        if screen_gray is None:
            return "error"
        
        # 多线程并行匹配
        futures = []
        self.best_scores.clear()
        for state in profile["Templates"]:
            future = self.executor.submit(
                self._parallel_match, state, screen_gray
            )
            futures.append(future)
        
        # 等待所有任务完成
        for future in futures:
            future.result()
        
        # 批量更新配置
        self._update_profile()
        
        # 快速决策
        best_state = max(self.best_scores, key=self.best_scores.get)
        if self.best_scores.get("result_screen", 0) > 0.9:
            best_state = "result_screen"
        
        return best_state
