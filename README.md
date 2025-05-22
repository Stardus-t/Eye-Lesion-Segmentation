Eye-Lesion-Segmentation（眼科病变区域分割系统）

项目概述

本项目致力于开发一个高精度的眼科医学图像分割网站，目前利用 Vision Transformer 架构自动识别和分割眼底图像中的病变区域。系统支持从数据预处理、模型训练到临床诊断的全流程，并提供直观的可视化界面辅助医生决策。

应用场景
糖尿病视网膜病变早期筛查
青光眼相关病变识别
眼科临床诊断辅助工具

安装依赖

pip install torch torchvision matplotlib tqdm  scikit-image opencv-python json dotenv


配置数据路径：
bash
# .env文件配置
ANNO_K_PATH=D:\Py\EYE\Data\anno_K\roi_jl250105.json

ANNO_IMAGES_PATH=D:\Py\EYE\Data\anno_images



# 训练模型（支持断点继续训练）
python train.py --epochs 50 --batch_size 8 --lr 0.0001 --resume checkpoint/model.raw.pth


# 对单张图像进行分割并可视化
python inference.py --image_path sample_images/eye_lesion.jpg --model_path checkpoint/best_model.pth

分割结果示例
![image](https://github.com/user-attachments/assets/65785dd0-fa11-45c0-965a-1072a15e5c18)


左：原始眼底图像 | 中：真实病变区域掩码 | 右：模型分割结果

# 网站演示
![image](https://github.com/user-attachments/assets/4ec28036-699e-4f6d-a6ad-6ff87063296a)


临床应用建议

本系统为辅助诊断工具，最终诊断应由专业眼科医生确认
建议对模型输出的高风险区域进行重点关注
可结合其他临床检查结果进行综合判断

