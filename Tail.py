#!/usr/bin/env python3
import json
import time
import os

with open("Data/json/profile.json", "r", encoding="utf-8") as f:
    profile = json.load(f)

STATE_FILE = profile['PATH']['GameStatePath']
ACTION_FILE = profile['PATH']['ActionPath']

def deep_diff(old, new, prefix=""):
    """
    递归比较两个数据结构（字典、列表及基本类型），返回列表形式的变化。
    每个元素为 (参数全名, 旧值, 新值)
    """
    changes = []
    if isinstance(old, dict) and isinstance(new, dict):
        keys = set(old.keys()) | set(new.keys())
        for key in keys:
            full_key = f"{prefix}.{key}" if prefix else key
            if key not in old:
                changes.append((full_key, None, new[key]))
            elif key not in new:
                changes.append((full_key, old[key], None))
            else:
                changes.extend(deep_diff(old[key], new[key], full_key))
    elif isinstance(old, list) and isinstance(new, list):
        # 若列表内容不相等，则视为变化，输出整个列表的变化
        if old != new:
            changes.append((prefix, old, new))
    else:
        if old != new:
            changes.append((prefix, old, new))
    return changes

def write_changes(changes):
    """将变化信息以追加的方式写入到 OUTPUT_FILE 文件中"""
    with open(ACTION_FILE, 'a', encoding='utf-8') as f:
        for param, old_val, new_val in changes:
            f.write(f"参数 '{param}' 变化：{old_val} --> {new_val}\n")
            

def monitor_json(filename):
    """ 监视 JSON 文件变化 """
    # 初始检查文件是否存在
    if not os.path.exists(filename):
        print(f"文件 {filename} 不存在。")
        return

    # 初始读取文件内容
    last_mtime = os.path.getmtime(filename)
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            prev_data = json.load(f)
    except Exception as e:
        print("初始读取 JSON 失败：", e)
        return

    print("开始监视文件变化……")
    while True:
        time.sleep(1)  # 每秒检查一次
        try:
            current_mtime = os.path.getmtime(filename)
        except Exception as e:
            continue  # 文件可能暂时不可用，跳过此次循环

        if current_mtime != last_mtime:
            last_mtime = current_mtime
            try:
                with open(filename, 'r', encoding='utf-8') as f:
                    curr_data = json.load(f)
            except Exception as e:
                print("读取 JSON 时发生错误：", e)
                continue

            changes = deep_diff(prev_data, curr_data)
            if changes:
                write_changes(changes)
            prev_data = curr_data

if __name__ == '__main__':
    monitor_json(STATE_FILE)
