# Eye-Lesion-Segmentation（眼科病变区域分割系统）
项目概述
本项目致力于开发一个高精度的眼科医学图像分割模型，利用 Vision Transformer 架构自动识别和分割眼底图像中的病变区域。系统支持从数据预处理、模型训练到临床诊断的全流程，并提供直观的可视化界面辅助医生决策。
应用场景
糖尿病视网膜病变早期筛查
青光眼相关病变识别
眼科临床诊断辅助工具
项目结构
plaintext
.
├── transforms.py         # 医学图像专用数据增强
├── data_preprocessing.py # 眼底图像标准化处理
├── train.py              # 模型训练与优化
├── inference.py          # 病变区域分割推理
├── visualization.py      # 分割结果可视化
├── metrics.py            # 医学分割评估指标
└── dataset.py            # 眼科图像数据集管理
技术亮点
ViT 架构：采用 Vision Transformer 提取全局上下文特征，提升病变边界分割精度
混合损失函数：结合 Dice Loss 和 BCE Loss 优化医学图像分割任务
数据增强策略：针对眼科图像特点设计的旋转、缩放、弹性变换等增强方法
安装依赖
bash
pip install torch torchvision matplotlib tqdm  scikit-image opencv-python json dotenv
数据准备
将眼底图像数据集按以下结构组织：
plaintext
dataset/
├── train/
│   ├── images/
│   └── masks/
└── test/
    ├── images/
    └── masks/

配置数据路径：
bash
# .env文件配置
ANNO_K_PATH=D:\Py\EYE\Data\anno_K\roi_jl250105.json
ANNO_IMAGES_PATH=D:\Py\EYE\Data\anno_images

模型训练
bash
# 训练模型（支持断点续训）
python train.py --epochs 50 --batch_size 8 --lr 0.0001 --resume checkpoint/best_model.pth
病变区域分割推理
bash
# 对单张图像进行分割并可视化
python inference.py --image_path sample_images/eye_lesion.jpg --model_path checkpoint/best_model.pth
可视化示例

分割结果示例
![image](https://github.com/user-attachments/assets/f4f90612-22e1-4dab-9c0e-e157727a2931)

左：原始眼底图像 | 中：真实病变区域掩码 | 右：模型分割结果

临床应用建议
本系统为辅助诊断工具，最终诊断应由专业眼科医生确认
建议对模型输出的高风险区域进行重点关注
可结合其他临床检查结果进行综合判断

