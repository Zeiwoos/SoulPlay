import pygetwindow as gw

# 获取所有窗口标题
windows = gw.getAllTitles()

# 查找包含 "雀魂" 的窗口
majsoul_windows = [title for title in windows if "雀魂" in title]

print("雀魂麻将窗口:", majsoul_windows)
