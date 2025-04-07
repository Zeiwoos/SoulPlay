import pyautogui
import time
import os
import threading
from datetime import datetime
import json
import pygetwindow as gw
import sys
import ctypes
import queue
from PIL import Image
from ImageProcess import ImageDetection,ImageProcessor
from GameRunStateTest import GameRunStateDetector

# 预加载配置
with open("Data/json/profile.json", "r", encoding="utf-8") as f:
    profile = json.load(f)

class HighQualityCapturer:
    def __init__(self):
        # 硬件加速配置
        ctypes.windll.shcore.SetProcessDpiAwareness(2) if hasattr(ctypes.windll, 'shcore') else ctypes.windll.user32.SetProcessDPIAware()
        
        # 配置参数
        self.cfg = {
            'interval': profile['ScreenShotInterval'],
            'output_dir': profile['PATH']['ScreenShotPath'],
            'max_files': profile['MaxScreenShotCount'],  # 限制最大文件数
            'game_title': profile['GameWindowTitle_CN'],
            'retry_limit': profile['Retry_Count']
        }
        
        # 状态控制
        self.running = False
        self.capture_thread = None
        self.isWindowActive = True  # 窗口激活状态
        self.process_running = True  # 独立控制处理线程
        self.window_cache = {'last_check': 0, 'region': None}
        self.task_queue = queue.Queue(maxsize=profile['MaxQueueCount'])  # 控制内存占用
        
        # 预加载资源
        os.makedirs(self.cfg['output_dir'], exist_ok=True)
        self.detector = GameRunStateDetector()
        self.process_thread = threading.Thread(target=self._process_worker, daemon=True)
        self.ImageProcessor = ImageProcessor()
        
        # 性能计数器
        self.counter = {
            'total': self._init_file_counter(),
            'start_time': time.time(),
            'last_cleanup': 0
        }

    def _init_file_counter(self)-> int:
        """优化文件计数器初始化"""
        try:
            return len([f for f in os.listdir(self.cfg['output_dir']) if f.endswith('.png')])
        except:
            return 0

    def _get_window_region(self) -> tuple[tuple[int, int, int, int], object]:
        """带缓存的窗口区域获取"""
        now = time.time()
        if now - self.window_cache['last_check'] > 1.0:  # 降低检查频率
            try:
                windows = gw.getWindowsWithTitle(self.cfg['game_title'])
                if windows:
                    win = windows[0]
                    if win.isActive and not win.isMinimized:
                        self.window_cache = {
                            'region': (win.left+5, win.top+40, win.width-10, win.height-50),
                            'last_check': now,
                            'window': win
                        }
            except Exception as e:
                print(f"⚠️ 窗口检测异常: {str(e)}")
        return self.window_cache.get('region'), self.window_cache.get('window')

    def _capture_image(self)-> str:
        """高质量截图方法"""
        try:
            region, window = self._get_window_region()
            if not region or not window.isActive or not window:
                if self.isWindowActive:
                    self.isWindowActive = False
                    print("🚨窗口未激活，跳过截图")
                    return None
                else:
                    # 如果窗口未激活且之前已激活，则不打印警告
                    return None
            else:
                self.isWindowActive = True
            
            # 使用更快的内存映射方式
            img = pyautogui.screenshot(region=region)
            
            # 生成唯一文件名
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
            filename = f"game_{timestamp}.png"
            filepath = os.path.join(self.cfg['output_dir'], filename)
            
            # 无损保存（压缩级别0）
            img.save(filepath, format='PNG', compress_level=0)
            
            return filepath
        
        except Exception as e:
            print(f"📸 截图失败: {str(e)}")
            return None

    def _process_worker(self)-> None:
        """修改后的处理线程"""
        while self.process_running:  # 使用独立控制变量
            try:
                filepath = self.task_queue.get(timeout=1)
                GameState = self.detector.get_game_state(filepath)
                if GameState == "GameStart" or GameState == "GameRunning":
                    self.detector.GameStateUseful = ImageDetection(filepath, self.ImageProcessor, GameState)
                if GameState == "GameEnd":
                    # 处理游戏结束状态
                    BoardState = {'state':"GameEnd"}
                    with open("./Data/json/board-state/BoardState.json", 'w', encoding='utf-8') as f:
                        json.dump(BoardState, f, indent=2, ensure_ascii=False)

                self.task_queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                print(f"处理失败: {e}")


    def _auto_cleanup(self)-> None:
        """优化清理逻辑"""
        try:
            files = sorted(os.listdir(self.cfg['output_dir']),
                          key=lambda f: os.path.getctime(os.path.join(self.cfg['output_dir'], f)))
            while len(files) > self.cfg['max_files']:
                os.remove(os.path.join(self.cfg['output_dir'], files.pop(0)))
        except Exception as e:
            print(f"⚠️ 清理失败: {str(e)}")

    def _precision_capture_loop(self)-> None:
        """精准间隔捕获循环"""
        next_time = time.time()
        while self.running:
            # 执行捕获
            filepath = self._capture_image()
            if filepath:
                # 增加文件校验逻辑
                if os.path.exists(filepath) and os.path.getsize(filepath) > 0:
                    print(f"📸 截图成功: {filepath}")
                    try:
                        self.task_queue.put_nowait((filepath))
                    except queue.Full:
                        print("⚠️ 任务队列已满，跳过处理")
                else:
                    print(f"❌ 截图文件 {filepath} 未正确生成")
                
                # 定期清理
                if time.time() - self.counter['last_cleanup'] > 60:
                    self._auto_cleanup()
                    self.counter['last_cleanup'] = time.time()

            # 精准间隔控制
            next_time += self.cfg['interval']
            sleep_time = max(0, next_time - time.time())
            if sleep_time > 0:
                time.sleep(sleep_time)
            else:
                next_time = time.time()  # 补偿超时

    def start(self)-> None:
        """优化启动方法"""
        if not self.running:
            self.running = True
            self.process_running = True
            # 确保线程重新创建
            self.process_thread = threading.Thread(target=self._process_worker, daemon=True)
            self.capture_thread = threading.Thread(target=self._precision_capture_loop, daemon=True)
            self.process_thread.start()
            self.capture_thread.start()
            print(f"🚀 服务已启动 | 路径: {os.path.abspath(self.cfg['output_dir'])} | 间隔: {self.cfg['interval']}s")

    def stop(self)-> None:
        """优化停止方法"""
        if self.running:
            # 第一步：停止捕获线程
            self.running = False
            if self.capture_thread and self.capture_thread.is_alive():
                self.capture_thread.join(timeout=2)
            
            # 第二步：停止处理线程
            self.process_running = False
            if self.process_thread and self.process_thread.is_alive():
                self.process_thread.join(timeout=2)
            
            # 第三步：清空任务队列
            while not self.task_queue.empty():
                try:
                    self.task_queue.get_nowait()
                    self.task_queue.task_done()
                except queue.Empty:
                    break

            # 第四步：强制终止残留线程
            if self.capture_thread.is_alive() or self.process_thread.is_alive():
                print("⚠️ 检测到未正常退出的线程，强制终止中...")
                os._exit(1)  # 最后手段

            duration = time.time() - self.counter['start_time']
            print(f"""
            🛑 服务已停止
            📸 总截图数: {self.counter['total']}
            ⏳ 运行时长: {duration:.1f}s
            ⚡ 平均频率: {self.counter['total']/duration:.1f}fps
            """)