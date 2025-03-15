import cv2
import numpy as np
import uuid
import os

def get_mahjongs_contours(img):
    """
    获取麻将牌的轮廓。
    :param img: 输入的麻将桌面图像
    :return: 轮廓列表
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    thresh = np.zeros_like(gray)
    for i in range(19, 25, 2):
        thresh += cv2.inRange(img, (i * 10, i * 10, i * 10), (i * 10 + 20, i * 10 + 20, i * 10 + 20))
    
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    valid_boxes = []
    
    for i in range(len(contours)):
        area = cv2.contourArea(contours[i])
        if area < 300 or hierarchy[0][i][3] != -1:
            continue
        rect = cv2.minAreaRect(contours[i])
        box = cv2.boxPoints(rect)
        valid_boxes.append(np.int64(box))
    
    return valid_boxes

def extract_tiles(img):
    """
    从输入图像中提取麻将牌图像。
    :param img: 输入的麻将桌面图像
    :return: 切割出的麻将牌图像列表
    """
    boxes = get_mahjongs_contours(img)
    tiles = []
    padding = 1  # 可调整的边缘填充量
    
    for box in boxes:
        max_x = max(box[:, 1])
        min_x = min(box[:, 1])
        max_y = max(box[:, 0])
        min_y = min(box[:, 0])
        
        tile_img = img[min_x+padding:max_x-padding, min_y+padding:max_y-padding].copy()
        tiles.append(tile_img)
    
    return tiles

def process_folder(input_folder, output_folder):
    """
    处理文件夹下所有图片（包含子文件夹），并将结果保存到输出文件夹。
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    for root, _, files in os.walk(input_folder):
        for file in files:
            if file.lower().endswith(('png', 'jpg', 'jpeg')):
                img_path = os.path.join(root, file)
                img = cv2.imread(img_path)
                if img is None:
                    continue
                
                tiles = extract_tiles(img)
                
                # 创建对应的子文件夹
                subfolder_name = os.path.splitext(file)[0]
                subfolder_path = os.path.join(output_folder, subfolder_name)
                if not os.path.exists(subfolder_path):
                    os.makedirs(subfolder_path)
                
                # 确认牌数不为0
                if len(tiles) == 0:
                    continue

                # 保存切割出的麻将牌图像
                for i, tile in enumerate(tiles):
                    tile_path = os.path.join(subfolder_path, f'{i}.png')
                    cv2.imshow(f'Tile {i}', tile)
                    cv2.imwrite(tile_path, tile)
                print(f"处理完成: {file}, 生成 {len(tiles)} 张麻将牌。")

if __name__ == '__main__':
    input_folder = 'Data/recogition/output_first/'
    output_folder = 'Data/recogition/output_final/'
    process_folder(input_folder, output_folder)
