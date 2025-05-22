from flask import Flask, render_template, request, send_file
import torch
from PIL import Image
import numpy as np
from networks.vit_seg_modeling import VisionTransformer, CONFIGS
import io

app = Flask(__name__)

# 加载配置
config = CONFIGS['R50-ViT-B_16']

# 初始化模型
model = VisionTransformer(config, img_size=256, num_classes=config.n_classes, zero_head=False, vis=False)

# 加载预训练权重
weight_file = r'D:\Py\EYE\checkpoint\weight_raw.pth'
checkpoints = torch.load(weight_file, weights_only=True)
model.load_state_dict(checkpoints['model_state_dict'])
model.eval()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/segment', methods=['POST'])
def segment():
    # 获取上传的图像
    file = request.files['image']
    img = Image.open(file.stream)
    img = img.resize((256, 256))
    img = np.array(img)
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    img = img.unsqueeze(0)

    # 进行分割
    with torch.no_grad():
        output = model(img)
        preds = torch.sigmoid(output)
        preds = (preds > 0.5).float()

    # 将分割结果转换为图像
    mask = preds[0, 0].cpu().numpy() * 255
    mask = Image.fromarray(mask.astype(np.uint8))

    # 将分割结果保存到内存中
    buffer = io.BytesIO()
    mask.save(buffer, format='PNG')
    buffer.seek(0)

    return send_file(buffer, mimetype='image/png')

if __name__ == '__main__':
    app.run(debug=True)