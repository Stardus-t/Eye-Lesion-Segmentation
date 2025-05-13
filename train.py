import os
import torch
from networks.vit_seg_modeling import VisionTransformer,CONFIGS
from Data.local_dataset.dataset import train_loader
from tqdm import tqdm
import matplotlib

matplotlib.rcParams['font.sans-serif'] = ['SimHei']
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# 修改DiceLoss为适用于二分类的版本
class DiceLoss(torch.nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()
        self.smooth = 1e-5

    def forward(self, inputs, target):
        inputs = torch.sigmoid(inputs)  # 使用sigmoid替代softmax
        intersect = (inputs * target).sum()
        denominator = inputs.sum() + target.sum()
        dice = (2. * intersect + self.smooth) / (denominator + self.smooth)
        return 1 - dice

batch_size=8
num_epoch=140
device="cuda" if torch.cuda.is_available() else "cpu"
print(f'Now using {device} as device')

config = CONFIGS['R50-ViT-B_16']
model = VisionTransformer(config, img_size=256, num_classes=config.n_classes, zero_head=False, vis=False).to(device)
optimizer=torch.optim.SGD(model.parameters(),lr=0.01,momentum=0.9,weight_decay=0.0001)
ce_loss=torch.nn.BCEWithLogitsLoss()
dice_loss=DiceLoss()

weight_path= r'D:\Py\EYE\checkpoint'

if not os.path.exists(weight_path):
    os.makedirs(weight_path)

weight_file=os.path.join(weight_path,'weight_raw.pth')
if os.path.exists(weight_file):
     checkpoint=torch.load(weight_file,map_location=device,weights_only=True)
     model.load_state_dict(checkpoint['model_state_dict'])
     optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
     start_epoch=checkpoint['epoch']+1
     print(f'继续开始从第{start_epoch}次开始训练')
else:
     start_epoch=0
     print(f'未找到权重文件，从头开始训练')


max_iterations=num_epoch*len(train_loader)
base_lr=0.01
iter_num=0

Loss=[]
Iter_num=[]
for epoch in range(start_epoch + 1, num_epoch + 1):
    model.train()
    total_loss = 0.0

    # 使用tqdm包装train_loader，创建内层进度条
    progress_bar = tqdm(enumerate(train_loader), total=len(train_loader),
                        desc=f'Epoch {epoch}/{num_epoch}', position=0, leave=True)

    for i, (image, mask) in progress_bar:
        image, mask = image.to(device), mask.to(device)

        optimizer.zero_grad()
        output = model(image)
        loss_ce = ce_loss(output, mask.float())
        loss_dice = dice_loss(output, mask)
        loss = 0.5 * loss_ce + 0.5 * loss_dice
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        # 更新学习率
        lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr_

        iter_num += 1
        current_avg_loss = total_loss / (i + 1)  # 当前平均损失

        # 更新进度条显示的信息
        progress_bar.set_postfix({
            'Loss': f'{current_avg_loss:.4f}',
            'LR': f'{lr_:.6f}'
        })

        # 保存损失和迭代次数用于后续绘图
        Loss.append(current_avg_loss)
        Iter_num.append(iter_num)

    # 每个epoch结束后保存模型
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }
    weight_file = os.path.join(weight_path, 'weight_raw.pth')
    torch.save(checkpoint, weight_file)

    # 打印最终的epoch统计信息
    print(f'Epoch {epoch}/{num_epoch} 完成, 最终平均 Loss: {total_loss / len(train_loader):.4f}')


