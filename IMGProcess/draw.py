import cv2
def draw_regions(img, hand_regions, regions):
    for (key, x, y, w, h) in hand_regions:
        if key in regions:  # 确保 key 存在于 regions
            color = regions[key]['color']
        else:
            color = (255, 255, 255)  # 如果找不到，默认白色
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
    # img = resize_for_display(img)
    # cv2.imshow('img', img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

def draw_original_regions(img, regions):
    for key, region in regions.items():
        cv2.rectangle(img, (region['rect'][0], region['rect'][1]), (region['rect'][2], region['rect'][3]), region['color'], 2)
    img = resize_for_display(img)
    cv2.imshow('img', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def resize_for_display(image, max_width=1000, max_height=800):
    h, w = image.shape[:2]
    scale = min(max_width / w, max_height / h)  # 计算缩放比例
    if scale < 1:  # 仅当图片过大时缩小
        image = cv2.resize(image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    return image

    