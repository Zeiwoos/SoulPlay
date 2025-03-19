import pyautogui
import time
import os
import cv2
import threading
from datetime import datetime
import json
import pygetwindow as gw
import sys
import ctypes
from ImageProcess import ImageDetection
from GameRunStateTest import get_game_state
from IMGProcess.TileStateGenerater import delete_folders
# 读取配置文件
with open("Data/json/profile.json", "r", encoding="utf-8") as f:
    profile = json.load(f)

class GameScreenCapturer:
    def __init__(self):
        self.capture_interval = profile['ScreenShotInterval']
        self.output_dir = profile['PATH']['ScreenShotPath']
        self.running = False
        self.capture_thread = None
        self.max_files = 1000
        self.retry_interval = profile['Retry_Interval']
        self.game_title = profile['GameWindowTitle_CN']

        # DPI 适配
        self._set_dpi_awareness()
        
        # 初始化目录
        os.makedirs(self.output_dir, exist_ok=True)
        self.image_counter = self._get_initial_counter()
        self.total_captured = 0
        self.start_time = time.time()

    def _set_dpi_awareness(self):
        """设置 DPI 适配"""
        try:
            ctypes.windll.shcore.SetProcessDpiAwareness(2)
        except:
            ctypes.windll.user32.SetProcessDPIAware()

    def _get_initial_counter(self):
        """统计已有截图文件数"""
        try:
            return len([f for f in os.listdir(self.output_dir) if f.startswith('game_') and f.endswith('.png')])
        except FileNotFoundError:
            return 0

    def _get_game_region(self):
        """获取游戏窗口的坐标区域"""
        # 获取雀魂麻将窗口
        windows = gw.getWindowsWithTitle(self.game_title)
        if not windows:
            raise RuntimeError(f"未找到窗口: {self.game_title}")

        # 取第一个匹配的窗口
        game_window = windows[0]
        if game_window.isMinimized:
            raise RuntimeError(f"游戏窗口已最小化: {self.game_title}")

        # 获取窗口坐标
        x, y, width, height = game_window.left, game_window.top, game_window.width, game_window.height

        return (x+5, y+40, width-10, height-50)

    def _get_filename(self):
        """生成时间戳文件名"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        return [os.path.join(self.output_dir, f"game_{timestamp}.png"), f"game_{timestamp}.png"]

    def _auto_cleanup(self):
        """自动清理旧截图"""
        try:
            files = [os.path.join(self.output_dir, f) for f in os.listdir(self.output_dir)]
            files.sort(key=os.path.getctime)
            while len(files) > self.max_files:
                os.remove(files.pop(0))
                self.image_counter -= 1
        except Exception as e:
            print(f"⚠️ 清理失败: {str(e)}")

    def _capture_loop(self):
        """截图主循环"""
        last_capture = time.time()
        retry_count = 0

        while self.running:
            try:
                # 获取游戏窗口
                windows = gw.getWindowsWithTitle(self.game_title)
                if not windows:
                    raise RuntimeError(f"⚠️ 未找到窗口: {self.game_title}")

                game_window = windows[0]

                # 检查窗口是否在最上方
                if not game_window.isActive:
                    print("🚨窗口未激活，跳过截图")
                    time.sleep(self.capture_interval)
                    continue

                # 获取游戏窗口区域
                region = self._get_game_region()

                # 执行截图
                filePath, filename = self._get_filename()
                screenshot = pyautogui.screenshot(region=region)
                screenshot.save(filePath, compress_level=1)
                print(f"✅ 截图成功:{filePath}")

                if get_game_state(filePath) == "in_game":
                    # 进行图片处理
                    ImageDetection(filename)

                # 计算精确的等待时间
                current_time = time.time()
                sleep_time = max(0, last_capture + self.capture_interval - current_time)
                time.sleep(sleep_time)

                # 更新状态
                last_capture = time.time()
                self.image_counter += 1
                self.total_captured += 1
                retry_count = 0

                # 定期清理
                if self.image_counter % 50 == 0:
                    self._auto_cleanup()

            except RuntimeError as e:
                print(f"窗口异常: {str(e)}，重试中... ({retry_count+1}/{profile['Retry_Count']})")
                retry_count += 1
                time.sleep(self.retry_interval)

            except Exception as e:
                print(f"未知错误: {str(e)}")
                time.sleep(1)


    def start(self):
        """启动截图服务"""
        if not self.running:
            self.running = True
            self.capture_thread = threading.Thread(target=self._capture_loop)
            self.capture_thread.daemon = True
            self.capture_thread.start()
            print(f"📸 截图服务已启动\n📂 存储路径: {os.path.abspath(self.output_dir)}\n⏳ 间隔: {self.capture_interval}s")


    def stop(self):
        """停止截图服务"""
        if self.running:
            self.running = False
            print("🛑 正在停止截图服务...")
            self.capture_thread.join(timeout=2)
            duration = time.time() - self.start_time
            print(f"✅ 服务已停止\n📸 总截图数: {self.total_captured}\n⏳ 运行时长: {duration:.1f}s\n⚡ 平均频率: {self.total_captured/duration:.1f}fps" if self.total_captured else "⚡ 无截图数据")
            
