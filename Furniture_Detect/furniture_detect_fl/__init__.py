'''
set FLASK_APP=furniture_detect_fl
set DEBUG=True
flask run -h 192.~~~

'''

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import datasets,models,transforms
from PIL import Image
import cv2
import numpy as np
from glob import glob

import flask
from flask import Flask,request,jsonify

import requests
import base64

import os
import shutil

def create_app():
    app = Flask(__name__)

    @app.route('/')
    def index():
        return '안녕하세요'

    @app.route('/De_Furniture',methods=['POST'])
    def Detect_furniture():
        class_names = ["원형테이블", "테이블"]

        transforms_test = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(device)

        # ddd 모델 불러오기
        model_yolo = torch.hub.load('ddd', 'custom',
                                    path='C:/Users/HP/Desktop/LastProject/Furniture_Detect/furniture_last.pt',
                                    source='local')
        model_yolo.conf = 0.4

        # resnet 모델 불러오기
        model_res = torch.load('C:/Users/HP/Desktop/LastProject/Furniture_Detect/furniture_2.pth',
                               map_location=torch.device('cpu'))
        model_res.eval()
        model_res = model_res.to(device)

        print('=' * 100)
        file = request.files['img']
        print(file)
        file.save('./furniture_detect_fl/sample_interior.jpg')

        result = model_yolo('./furniture_detect_fl/sample_interior.jpg')
        print(result.pandas().xyxy[0])

        result.crop()

        # crop된 이미지가 여럿일때
        # path = 'C:/Users/HP/Desktop/LastProject/Furniture_Detect/runs/detect/exp/crops'
        #
        # print(glob(path+'/*/*.jpg'))
        #
        # return_list = []
        # for furniture_file in glob(path+'/*/*.jpg'):
        #     image = Image.open(furniture_file)
        #     image = transforms_test(image).unsqueeze(0).to(device)
        #
        #     with torch.no_grad():
        #         outputs = model_res(image)
        #
        #         _, preds = torch.max(outputs, 1)
        #
        #         # print(preds)
        #
        #         return_list.append(class_names[preds[0]])
        #         print(class_names[preds[0]])

        image = Image.open('./runs/detect/exp/crops/Table/sample_interior.jpg')
        image = transforms_test(image).unsqueeze(0).to(device)
        with torch.no_grad():
            outputs = model_res(image)

            _, preds = torch.max(outputs, 1)

            # print(preds)

            print(class_names[preds[0]])


        dir_path = './runs'

        print(os.path.exists(dir_path))
        if os.path.exists(dir_path):
            shutil.rmtree(dir_path)

        print('=' * 100)
        # print(return_list)
        return class_names[preds[0]]

    return app