import torch
import clip
from PIL import Image
import os
import pandas as pd
import numpy as np

from flask import Flask,request, jsonify
import urllib.request

import faiss

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

print(device)



def clip_get_vector(image_path):
    image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)

    # 이미지 임배딩 벡터값(특징 추출)
    with torch.no_grad():
        image_features = model.encode_image(image)

    return image_features[0]

a = clip_get_vector('C:/Users/HP/Desktop/LastProject/last/app/2_save_crop_url_image/다운로드(2).jpg')
print(a.cpu().numpy().tolist())