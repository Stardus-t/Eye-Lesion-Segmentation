import matplotlib.pyplot as plt
import torch
from Data.local_dataset.dataset import test_loader
from networks.vit_seg_modeling import VisionTransformer,CONFIGS
plt.rcParams["font.family"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

device='cuda'if torch.cuda.is_available() else 'cpu'
print(f'Now using {device} as device')

config = CONFIGS['R50-ViT-B_16']
model = VisionTransformer(config, img_size=256, num_classes=config.n_classes, zero_head=False, vis=False).to(device)
weight_file=r'D:\Py\EYE\checkpoint\weight_raw.pth'
checkpoints=torch.load(weight_file,weights_only=True)
model.load_state_dict(checkpoints['model_state_dict'])

model.eval()
ACC=0
IOU=0
with torch.no_grad():
    for image,mask in test_loader:
        image,mask=image.to(device),mask.to(device)
        output=model(image)
        preds=torch.sigmoid(output)
        preds=(preds>0.5).float()

        correct=(preds==mask).sum()
        acc=correct/mask.numel()
        ACC+=acc

        insection=(mask*preds).sum()
        union=mask.sum()+preds.sum()-insection
        iou=insection/union
        IOU+=iou

        image_vis=image[0].cpu().permute(1,2,0).numpy()
        mask_vis=mask[0].cpu().numpy()
        pred_vis = preds[0].cpu().numpy()

        mask_vis=mask_vis.squeeze(0)
        pred_vis=pred_vis.squeeze(0)
        plt.figure(figsize=(10,8))
        plt.subplot(1,3,1)
        plt.imshow(image_vis)
        plt.axis('off')
        plt.title('原始图像')

        plt.subplot(1, 3, 2)
        plt.imshow(mask_vis,cmap='gray')
        plt.axis('off')
        plt.title('分割掩码')

        plt.subplot(1, 3, 3)
        plt.imshow(image_vis)
        plt.imshow(mask_vis,alpha=0.6,cmap='Reds_r')
        plt.axis('off')
        plt.title('掩码叠加')

        plt.show()
    print(f'准确率为：{ACC/len(test_loader)*100}%')
    print(f'IoU为：{IOU/len(test_loader)*100}%')