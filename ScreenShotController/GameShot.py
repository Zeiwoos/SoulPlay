import pyautogui
import time
import os
import threading
from datetime import datetime
import json

with open("Data/json/profile.json", "r", encoding="utf-8") as f:
    profile = json.load(f)
class GameScreenCapturer:
    def __init__(self):
        self.capture_interval = profile['ScreenShotInterval']  # 默认截图间隔（秒）
        self.output_dir = profile['PATH']['ScreenShotPath']
        self.running = False
        self.capture_thread = None
        self.region = None  # (left, top, width, height)
        self.image_counter = 1
        self.max_files = 1000  # 最大保存文件数
        
        # 自动创建输出目录
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 初始化性能计数器
        self.total_captured = 0
        self.start_time = time.time()

    def configure(self, interval=1.0, output_dir=None, region=None):
        """配置捕获参数"""
        self.capture_interval = max(0.1, interval)
        if output_dir:
            self.output_dir = output_dir
            os.makedirs(output_dir, exist_ok=True)
        self.region = region  # 截取区域 (left, top, width, height)

    def _get_filename(self):
        """生成带时间戳的文件名"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        return os.path.join(self.output_dir, f"game_{timestamp}.png")

    def _auto_cleanup(self):
        """自动清理旧文件"""
        if self.max_files <= 0:
            return
            
        files = sorted(os.listdir(self.output_dir), 
                      key=lambda x: os.path.getmtime(os.path.join(self.output_dir, x)))
        while len(files) > self.max_files:
            os.remove(os.path.join(self.output_dir, files[0]))
            files.pop(0)

    def _capture_loop(self):
        """截图循环核心逻辑"""
        last_capture = time.time()
        while self.running:
            try:
                # 计算精确等待时间
                elapsed = time.time() - last_capture
                sleep_time = max(0, self.capture_interval - elapsed)
                time.sleep(sleep_time)
                
                # 执行截图
                filename = self._get_filename()
                if self.region:
                    screenshot = pyautogui.screenshot(region=self.region)
                else:
                    screenshot = pyautogui.screenshot()
                
                screenshot.save(filename)
                self.image_counter += 1
                last_capture = time.time()
                self.total_captured += 1
                
                # 定期清理文件
                if self.image_counter % 100 == 0:
                    self._auto_cleanup()

            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"截图失败: {str(e)}")
                time.sleep(1)  # 错误后等待1秒

    def start(self):
        """启动截图线程"""
        if not self.running:
            self.running = True
            self.capture_thread = threading.Thread(target=self._capture_loop)
            self.capture_thread.daemon = True
            self.capture_thread.start()
            print(f"截图服务已启动，间隔：{self.capture_interval}s")
            
    def stop(self):
        """停止截图服务"""
        if self.running:
            self.running = False
            self.capture_thread.join()
            duration = time.time() - self.start_time
            print(f"截图服务已停止，共捕获 {self.total_captured} 张 (平均 {self.total_captured/duration:.1f}fps)")

    def get_recent_screenshot(self, count=1):
        """获取最新截图路径"""
        files = sorted(os.listdir(self.output_dir), 
                      key=lambda x: os.path.getmtime(os.path.join(self.output_dir, x)),
                      reverse=True)
        return [os.path.join(self.output_dir, f) for f in files[:count]]