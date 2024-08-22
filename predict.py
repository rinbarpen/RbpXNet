import torch
import wandb

def predict_one(model, input):
  model, input = model.to(wandb.config.device), input.to(wandb.config.device)
  
  model.eval()
  with torch.no_grad():
    predict = model(input)
  return predict
