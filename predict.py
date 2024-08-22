import torch

def predict_one(model, input):
  model.eval()
  with torch.no_grad():
    predicts = model(input)
  output = torch.argmax(predicts, dim=1)
  return output
  