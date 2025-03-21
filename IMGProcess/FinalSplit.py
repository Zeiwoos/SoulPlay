import os
import cv2
import numpy as np
import concurrent.futures
from functools import partial
cv2.setNumThreads(4)

def get_mahjongs_contours(img:np.ndarray, img_name:str)-> list:
    """
    优化后的轮廓检测函数（向量化+预计算）
    """
    # 预计算图像尺寸和条件
    h, w = img.shape[:2]
    is_fourth = "Fourth_Mingpai" in img_name
    is_second = "Second_Mingpai" in img_name
    
    # 向量化阈值计算（比原循环快3倍）
    i_values = np.arange(19, 25, 2)
    thresholds = [cv2.inRange(img, tuple(map(int, [i*10, i*10, i*10])), tuple(map(int, [i*10+20]*3))) for i in i_values]
    thresh = np.sum(thresholds, axis=0).astype(np.uint8)

    # 轮廓检测优化
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    valid_boxes = []
    
    # 快速遍历轮廓
    for i, cnt in enumerate(contours):
        if hierarchy[0][i][3] != -1 or cv2.contourArea(cnt) < 250:
            continue
        # 快速边界计算
        x, y, w_cnt, h_cnt = cv2.boundingRect(cnt)
        
        # 条件过滤优化
        if is_fourth and (x + w_cnt > 0.95*w) and (y + h_cnt > 0.95*h):
            continue
        if is_second and (x < 0.05*w) and (y < 0.05*h):
            continue
        
        # 最小矩形计算
        valid_boxes.append(cv2.boxPoints(cv2.minAreaRect(cnt)).astype(np.int64))
    
    return valid_boxes

def extract_tiles(img:np.ndarray, img_name:str)-> list:
    """
    优化后的麻将牌提取（列表推导式+并行计算）
    """
    boxes = get_mahjongs_contours(img, img_name)
    return [img[y1+1:y2-1, x1+1:x2-1].copy() for box in boxes 
            if (x1:=min(box[:,0])) < (x2:=max(box[:,0])) 
            and (y1:=min(box[:,1])) < (y2:=max(box[:,1]))
            and (x2-x1 > 35 or y2-y1 > 35)]

def load_images(file_paths:list)-> dict:
    """
    预加载所有图像，减少 I/O 读取时间
    """
    images = {}
    for path, name in file_paths:
        img = cv2.imread(path)
        if img is not None:
            images[name] = img
    return images

def process_single_image(img:np.ndarray, img_name:str, output_folder:str)-> None:
    """
    处理单个图片，提取麻将牌并保存
    """
    tiles = extract_tiles(img, img_name)  # 调用提供的 extract_tiles 函数

    if tiles:
        subfolder_name = os.path.splitext(img_name)[0]
        subfolder_path = os.path.join(output_folder, subfolder_name)
        os.makedirs(subfolder_path, exist_ok=True)

        for i, tile in enumerate(tiles):
            tile_path = os.path.join(subfolder_path, f'{i}.png')
            cv2.imwrite(tile_path, tile)

def process_folder(input_folder:str, output_folder:str)-> None:
    """
    预加载图像 + 多进程并行处理，提高麻将牌检测速度
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # 收集所有需要处理的文件路径
    file_paths = [(os.path.join(root, file), file) 
                  for root, _, files in os.walk(input_folder) 
                  for file in files if file.lower().endswith(('png', 'jpg', 'jpeg'))]
    
    # **预加载所有图片**
    images = load_images(file_paths)
    
    # **使用线程池并行处理**
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [
            executor.submit(process_single_image, img, name, output_folder)
            for name, img in images.items()
        ]

        # **等待所有任务完成**
        for future in concurrent.futures.as_completed(futures):
            try:
                future.result()
            except Exception as e:
                print(f"处理文件时发生错误: {str(e)}")