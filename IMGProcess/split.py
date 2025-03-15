import cv2
import numpy as np
import uuid

def get_mahjongs_contours(img):
    """
    获取麻将牌的轮廓。
    :param img: 输入的麻将桌面图像
    :return: 轮廓列表
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 使用Canny边缘检测
    edges = cv2.Canny(gray, 50, 150)
    cv2.imshow('Edges', edges)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

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
        print(f"Tile size: {tile_img.shape}")  # 打印图像尺寸
        tiles.append(tile_img)
        
    return tiles

if __name__ == '__main__':
    # 改为对文件夹操作
    img = cv2.imread('Data\\recogition\output_first\lx\\fourth_Mingpai_lx.jpg')  # 请替换为实际图片路径
    tiles = extract_tiles(img)
    print(len(tiles))
    for i, tile in enumerate(tiles):
        cv2.imshow(f'Tile {i}', tile)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    # 保存切割出的麻将牌图像
    # for i, tile in enumerate(tiles):
    #     tile_path = f'Data/recogition/output/lx/tile_{uuid.uuid1()}.png'
    #     cv2.imwrite(tile_path, tile)
    print(f"麻将牌分割完成，共 {len(tiles)} 张麻将牌。")
