import os
import sys
import json
import time
import psutil
import threading
import shutil
import pygetwindow as gw
from concurrent.futures import ThreadPoolExecutor
from ScreenShot import HighQualityCapturer

sys.stdout.reconfigure(encoding="utf-8")

# 🌟 预加载配置
with open("Data/json/profile.json", "r", encoding="utf-8") as f:
    profile = json.load(f)

# 检测路径是否存在
def check_path(paths):
    for key, path in paths.items():
        if not os.path.exists(path):
            print(f"⚠️  路径 {path} 不存在，正在创建...")
            os.makedirs(path, exist_ok=True)
        else:
            print(f"✅ 路径 {path} 存在")


# **优化后的清空文件夹**
def clear_folders():
    """并行删除 ScreenShotPath、Split_FinalPath、Split_FirstPath 目录下的所有文件"""
    path_list = ["ScreenShotPath", "Split_FinalPath", "Split_FirstPath"]
    
    def delete_folder_contents(target_dir):
        if not os.path.exists(target_dir):
            return
        try:
            for entry in os.listdir(target_dir):
                full_path = os.path.join(target_dir, entry)
                if os.path.isfile(full_path) or os.path.islink(full_path):
                    os.remove(full_path)
                elif os.path.isdir(full_path):
                    shutil.rmtree(full_path)
            print(f"✅ {target_dir} 清理完成")
        except Exception as e:
            print(f"❌ 清理 {target_dir} 失败，错误：{e}")

    with ThreadPoolExecutor() as executor:
        for folder in path_list:
            executor.submit(delete_folder_contents, profile["PATH"].get(folder, ""))


class OptimizedGameMonitor:
    def __init__(self, capturer):
        self.capturer = capturer
        self.game_name = profile["game_name"]
        self.thread_pool = ThreadPoolExecutor(max_workers=2)
        self.last_state = False
        self.running = True
        self.cache = {
            "process_check": 0,
            "window_state": False,
        }

    def _check_process(self):
        """优化后的进程检测（带缓存）"""
        now = time.time()
        if now - self.cache["process_check"] > 2:  # 2秒缓存
            self.cache["process_check"] = now
            try:
                for p in psutil.process_iter(attrs=["name"]):
                    if p.info["name"] and self.game_name.lower() in p.info["name"].lower():
                        return True
            except psutil.AccessDenied:
                print("⚠️ 无法访问进程列表")
            return False
        return self.last_state

    def _check_window_active(self):
        """窗口激活状态检测（更稳健）"""
        try:
            windows = gw.getWindowsWithTitle(self.game_name)
            return bool(windows and windows[0].isActive)
        except Exception as e:
            print(f"⚠️ 窗口检测失败：{e}")
            return False

    def monitor_loop(self):
        """优化后的监控循环"""
        while self.running:
            # 🌟 并行检测进程和窗口状态
            process_future = self.thread_pool.submit(self._check_process)
            window_future = self.thread_pool.submit(self._check_window_active)

            process_running = process_future.result()
            window_active = window_future.result()

            # 🌟 仅检测进程状态（不依赖窗口状态）
            game_active = process_running  

            if game_active != self.last_state:
                if game_active:
                    print("🎮 游戏进入活跃状态")
                    self.capturer.start()
                else:
                    print("⏸️ 游戏已关闭，停止截图")
                    self.capturer.stop()
                self.last_state = game_active
                profile["is_game_running"] = game_active

            time.sleep(2)  # 缩短检测间隔，优化响应速度

    def stop(self):
        """增强停止方法"""
        self.running = False
        self.thread_pool.shutdown(wait=False, cancel_futures=True)  # 立即终止检测任务
        if threading.current_thread() is not monitor_thread:
            monitor_thread.join(timeout=1)


def valueInit():
    """优化初始化流程"""
    global profile

    # 🌟 并行路径检查
    with ThreadPoolExecutor() as executor:
        executor.submit(check_path, profile["PATH"])
        executor.submit(check_path, profile["Templates"])

    # 🌟 快速清空目录
    clear_folders()

    # 🌟 内存中更新配置避免写文件
    profile.update(
        {
            "is_game_running": False,
            "BestMatchState": {k: 0 for k in ["main_menu", "in_game", "result_screen", "matching"]},
        }
    )


if __name__ == "__main__":
    # 🌟 快速初始化
    valueInit()

    # 🌟 初始化高性能截图器
    capturer = HighQualityCapturer()

    # 🌟 启动优化后的监控器
    monitor = OptimizedGameMonitor(capturer)
    monitor_thread = threading.Thread(target=monitor.monitor_loop, daemon=True)
    monitor_thread.start()

    try:
        # 🌟 低功耗等待循环
        while True:
            time.sleep(5)
            # 状态报告
            print(
                f"📊 当前状态 | 截图队列: {capturer.task_queue.qsize()} | 内存占用: {psutil.Process().memory_info().rss // 1024 // 1024}MB"
            )
    except KeyboardInterrupt:
        print("\n🔴 正在安全停止服务...")
        # 停止顺序优化
        monitor.stop()
        capturer.stop()
        # 强制退出机制
        for t in threading.enumerate():
            if t is not threading.main_thread():
                t.join(timeout=0.5)
        print("✅ 服务已安全停止")
        os._exit(0)  # 确保完全退出