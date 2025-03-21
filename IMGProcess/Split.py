import os
import cv2
import numpy as np

def save_cropped_regions(img:np.ndarray, hand_regions:dict, img_name:str, output_folder:str)-> str:
    """保存裁剪后的区域"""
    base_name = os.path.splitext(img_name)[0]
    save_path = os.path.join(output_folder, base_name)
    os.makedirs(save_path, exist_ok=True)
    h, w = img.shape[:2]
    for key, (x, y, w_, h_) in hand_regions.items():
        cropped = img[max(0,y-20):min(h,y+h_+20), max(0,x-20):min(w,x+w_+20)]
        cv2.imwrite(os.path.join(save_path, f"{base_name}_{key}.{img_name.split('.')[-1]}"), cropped)
    return save_path