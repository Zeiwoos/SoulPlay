import json
import time
import psutil
import threading
from ImageProcess import check_path
from ScreenShotController.ScreenShot import GameScreenCapturer
from IMGProcess.TileStateGenerater import delete_folders
import sys
sys.stdout.reconfigure(encoding='utf-8')


with open("Data/json/profile.json", "r", encoding="utf-8") as f:
    profile = json.load(f)

def is_game_running(game_name, capturer):
    """检查游戏进程是否在运行，并控制截图"""
    global profile
    game_was_running = False  # 记录上一次游戏是否运行
    while True:
        running = False
        for process in psutil.process_iter(attrs=["pid", "name"]):
            if game_name.lower() in process.info["name"].lower():
                running = True
                break

        profile["is_game_running"] = running

        # **游戏状态变化时，启动或停止截图**
        if running and not game_was_running:
            print(f"🔵 游戏 {game_name} 运行中，开始截图...")
            capturer.start()
        elif not running and game_was_running:
            print(f"🔴 游戏 {game_name} 已关闭，停止截图...")
            capturer.stop()

        game_was_running = running  # 更新状态
        time.sleep(5)  # **每 5 秒检测一次**

def valueInit():
    """
    初始化配置
    """
    global profile
    # 检查路径是否存在
    check_path(profile['PATH'])
    check_path(profile['Templates'])
    # 删除SceenShotPath、Split_FinalPath、Split_FirstPath下所有文件
    delete_folders()
    # 初始化配置
    with open("Data/json/profile.json", "w", encoding="utf-8") as f:
        profile["is_game_running"] = False
        profile['BestMatchState']['main_menu'] = 0
        profile['BestMatchState']['in_game'] = 0
        profile['BestMatchState']['result_screen'] = 0
        profile['BestMatchState']['matching'] = 0
        json.dump(profile, f, ensure_ascii=False, indent=4)


if __name__ == '__main__':
    valueInit()
    # 初始化配置
    capturer = GameScreenCapturer()
    # 启动游戏检测线程
    game_monitor_thread = threading.Thread(target=is_game_running(profile["game_name"], capturer), daemon=True)
    game_monitor_thread.start()
    # 主线程保持运行，防止进程退出
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("🔴 退出程序...")