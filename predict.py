import torch
import wandb

def predict_one(model, input):
  model, input = model.to(wandb.config.device), input.to(wandb.config.device)
  
  model.eval()
  with torch.no_grad():
    predict = model(input)
    predict_np = predict.squeeze().cpu().detach().numpy()
    predict_np[predict_np >= 0.5] = 255
    predict_np[predict_np < 0.5] = 0
  return predict_np
