import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import datasets,models,transforms
from PIL import Image

import os
import shutil

class_names = ['원형테이블', '테이블']

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)

transforms_test = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

# ddd 모델 불러오기
model_yolo = torch.hub.load('../last/ddd', 'custom', path='C:/Users/HP/Desktop/LastProject/Furniture_Detect/furniture_last.pt', source='local')
model_yolo.conf = 0.4
result = model_yolo('./sample_images/sample_interior_image.jpg')
print(result.pandas().xyxy[0])

result.crop()

# resnet 모델 불러오기
model_res = torch.load('C:/Users/HP/Desktop/LastProject/Furniture_Detect/furniture_2.pth', map_location=torch.device('cpu'))
model_res.eval()

model_res = model_res.to(device)
image = Image.open('./runs/detect/exp/crops/Table/sample_interior_image.jpg')
image = transforms_test(image).unsqueeze(0).to(device)    # 이미지 불러와서 무조건! model에 맞게 변환해야함
# print(image.shape)

with torch.no_grad():
  outputs = model_res(image)

  _, preds = torch.max(outputs,1)

  print(preds)
  print(class_names[preds[0]])

dir_path = './runs'

print(os.path.exists(dir_path))
if os.path.exists(dir_path):

    shutil.rmtree(dir_path)