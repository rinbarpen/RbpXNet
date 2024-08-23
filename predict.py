import torch
import wandb
from PIL import Image
from matplotlib import pyplot as plt
from utils.utils import load_model
import numpy as np

def predict_one(model, input, device):
  model, input = model.to(device, dtype=torch.float32), input.to(device, dtype=torch.float32)
  
  model.eval()
  with torch.no_grad():
    predict = model(input)
    predict_np = predict.squeeze().cpu().detach().numpy()
    predict_np[predict_np >= 0.5] = 255
    predict_np[predict_np < 0.5] = 0
  return predict_np


def predict(net, input, device):
  net.load_state_dict(load_model(wandb.config.load))
  input = Image.open(input).convert('RGB')
  original_size = input.size
  input = input.resize((512,512))
  
  input = torch.from_numpy(np.array(input))
  input = input.expand(1, -1, -1, -1).permute(0, 3, 1, 2)
  
  output = predict_one(net, input, device)
  
  img = Image.fromarray(output, 'L')
  img = img.resize(original_size)
  img.save('./output/predict.png')
  
  # plt.imshow(np.array(input.squeeze(0).permute(1, 2, 0)), alpha=0.5)  # 原始图像
  plt.imshow(img)  # 叠加预测结果
  plt.axis('off')  # 不显示坐标轴
  plt.title('Model Prediction')
  plt.show()
