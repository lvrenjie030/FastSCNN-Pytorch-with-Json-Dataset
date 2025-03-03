# 本代码的作用是，根据XML文件中的坐标信息，生成对应的mask图像，这里采用png格式保存
import os
import numpy as np
import cv2
from pycocotools.coco import COCO
from pycocotools import mask as maskUtils

# COCO JSON 文件路径
coco_json_path = ""
image_dir = ""
output_mask_dir = ""

# 加载 COCO 数据
coco = COCO(coco_json_path)

# 获取所有图片 ID
image_ids = coco.getImgIds()

# 确保输出目录存在
os.makedirs(output_mask_dir, exist_ok=True)

for image_id in image_ids:
    img_info = coco.loadImgs(image_id)[0]
    width, height = img_info["width"], img_info["height"]

    # 创建一个空的掩码图像（单通道，初始化为0）
    mask = np.zeros((height, width), dtype=np.uint8)

    # 获取该图片对应的所有分割标注（多个类别）
    ann_ids = coco.getAnnIds(imgIds=image_id)
    anns = coco.loadAnns(ann_ids)

    for ann in anns:
        category_id = ann["category_id"]  # 类别 ID
        segmentation = ann["segmentation"]  # 分割数据

        if isinstance(segmentation, list):  # Polygon 格式
            for seg in segmentation:
                poly = np.array(seg, np.int32).reshape((-1, 2))
                cv2.fillPoly(mask, [poly], (255, 255, 255))  # 直接填充类别 ID

        elif isinstance(segmentation, dict):  # RLE 格式
            rle_mask = maskUtils.decode(segmentation)
            mask[rle_mask > 0] = category_id  # 赋值类别 ID

    # 保存为 PNG 掩码
    mask_path = os.path.join(output_mask_dir, f"{img_info['file_name'].replace('.jpg', '.png')}")
    cv2.imwrite(mask_path, mask)

print("转换完成！PNG 掩码已保存在:", output_mask_dir)
