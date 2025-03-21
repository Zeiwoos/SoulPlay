import os
import cv2
import numpy as np
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from IMGProcess.Classify import Classify

class BatchClassifier:
    def __init__(self):
        self.classifier = Classify()  # 初始化分类器

    def process_single_image(self, img_path):
        """处理单张图片（线程安全）"""
        filename = os.path.basename(img_path)
        try:
            img = cv2.imread(img_path)
            if img is None:
                return filename, "error: 无法读取图像"
            
            # 分类识别
            tile_name = self.classifier(img)
            return filename, tile_name
        except Exception as e:
            return filename, f"error: {str(e)}"

    def process_folder(self, input_folder:str, output_file="results.csv", max_workers=4)-> None:
        """
        多线程处理整个文件夹
        :param input_folder: 输入文件夹路径
        :param output_file:  输出结果文件路径
        :param max_workers:  最大线程数（建议设置为CPU核心数的2-4倍）
        """
        # 获取所有图片文件
        image_exts = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')
        image_files = [
            os.path.join(input_folder, f) 
            for f in os.listdir(input_folder)
            if f.lower().endswith(image_exts)
        ]
        
        # 创建线程池并处理
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # 提交所有任务
            futures = [executor.submit(self.process_single_image, path) for path in image_files]
            
            # 使用进度条监控处理进度
            results = []
            for future in tqdm(
                futures, 
                total=len(image_files), 
                desc="Processing Images", 
                unit="image"
            ):
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    print(f"处理任务时发生未捕获的异常: {str(e)}")

        # 写入结果（按文件名排序）
        results.sort(key=lambda x: x[0])  # 按文件名排序
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("filename,tile_name\n")
            for filename, tile_name in results:
                f.write(f"{filename},{tile_name}\n")
        print(f"处理完成！结果已保存至 {output_file}")



