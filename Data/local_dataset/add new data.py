import json
import os


# 假设新图片的信息
new_image_info = {
    "id": 100,
    "width": 362,
    "height":240,
    "file_name": "100.jpg"
}

new_roi_annotation = {
    "id": 99,
    "image_id": 100,
    "category_id": 1,
    "segmentation":[[176.67, 69.11, 170.61, 73.96, 166.45, 79.16, 161.08, 85.91, 156.4, 91.97, 153.98, 99.59, 152.59, 110.85, 152.42, 123.15, 155.02, 134.93, 157.44, 143.41, 158.66, 153.98, 165.58, 168.01, 172.17, 177.36, 183.25, 186.89, 196.59, 193.3, 206.63, 196.07, 220.14, 196.93, 231.92, 194.51, 248.55, 191.91, 257.04, 183.6, 270.03, 170.78, 283.19, 147.4, 285.79, 132.5, 285.62, 110.51, 282.15, 99.77, 277.82, 87.64, 269.16, 74.65, 255.82, 64.95, 238.33, 61.31, 210.79, 61.31, 188.79, 64.95, 185.5, 65.3]],
    "bbox": [152.42105263157893, 61.31483253588517, 133.3684210526316, 135.62009569377992],
    "area":14716.852304663364,
    "iscrowd": 0
}

pco_anno_data_dir = os.path.join(r"D:\Py\pythonProject\导师", "annotation2", "anno_K")
roi_path = os.path.join(pco_anno_data_dir, "roi_jl250105.json")
with open(roi_path, 'r') as f:
    roi_annos = json.load(f)

roi_annos['images'].append(new_image_info)

roi_annos['annotations'].append(new_roi_annotation)

with open(roi_path, 'w') as f:
    json.dump(roi_annos, f, indent=4)

print("数据已成功整合到 roi_jl250105.json 文件中。")