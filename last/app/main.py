from flask import Flask,request

import pandas as pd
import os
import csv
from glob import glob

import urllib.request

from PIL import Image
import io
import shutil

def image_to_byte_array(image_path):
    img = Image.open(image_path)
    imgByteArr = io.BytesIO()
    img.save(imgByteArr, format=img.format)
    imgByteArr = imgByteArr.getvalue()

    return imgByteArr


# # 채팅 필터 모델
# from Chat.Chat_Filter import filter_chatting

# 이미지 유사도 임베딩 모델
from .Image_similarity.Embed import get_vector
from .Image_similarity.Embed import image_to_vec

# 방 사진에서 가구 이미지 크롭
from .Detect_Furniture.img_detect import crop_model

# s3용
# from .config import BUCKET_NAME
# from .connection import s3_connection,s3_get_image_url

import boto3
from botocore.client import Config

'''
AWS_ACCESS_KEY = 
AWS_SECRET_KEY = 
BUCKET_NAME = 
'''


# def handle_upload_img(save_img_path, f):  # f = 파일명
#     data = open(save_img_path, 'rb')
#     # '로컬의 해당파일경로'+ 파일명 + 확장자
#
#     s3 = boto3.resource(
#         's3',
#         aws_access_key_id=AWS_ACCESS_KEY,
#         aws_secret_access_key=AWS_SECRET_KEY,
#         config=Config(signature_version='s3v4')
#     )
#     s3.Bucket(BUCKET_NAME).put_object(
#         Key=f, Body=data, ContentType='image/jpg')

def create_app():
    app = Flask(__name__)

    embedding_df = pd.DataFrame()

    @app.route('/')
    def index():
        return '안녕하세요'

    @app.route('/index')
    def reindex():
        return '안녕히 가세요'

    # @app.route('/add_image',methods=['POST'])
    # def add_images():
    #     img_file = request.files['img']
    #     save_img_path = f'./s3_images/{img_file.filename}'
    #     img_file.save(save_img_path)
    #
    #     handle_upload_img(save_img_path,img_file.filename)
    #     # return s3_get_image_url(s3,img_file.filename)
    #     return "success"


    # @app.route('/filter',methods=['POST'])
    # def fiLter():
    #     chat_text = request.json['CHAT']
    #     res_text = filter_chatting(chat_text)
    #
    #     return res_text
    
    # 이미지 올렸을 시 vector값 저장
    # @app.route('/add_furniture', methods=['POST'])
    # def Add_Furniture_Image():
    #     img_file = request.files['img']
    #
    #     save_img_path = f'./furniture_upload_folder/{img_file.filename}'
    #     img_file.save(save_img_path)
    #
    #     print("gggggggggggggggg")
    #     img_vec_df = pd.DataFrame(get_vector(save_img_path)).T
    #     img_vec_df['image'] = img_file.filename
    #     img_vec_df.rename(columns=lambda x: str(x), inplace=True)
    #
    #     with open('Image_similarity/Embedding_img.csv', mode='a', newline='') as f:
    #         csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    #         csv_writer.writerow(img_vec_df.iloc[0])
    #         f.close()
    #
    #     os.remove(save_img_path)
    #
    #     return "upload_vector_done!"

    # @app.route('/img_to_byte', methods=['POST'])
    # def Image_To_Byte():

    # 가구를 찾고 crop된 이미지를 s3에 저장하고 이미지 url return
    @app.route('/image_furniture_detect', methods=['POST'])
    def Furniture_Detecter():

        img_file = request.files['img']
        print(img_file)

        save_img_path = f'Detect_image_folder/{img_file.filename}'
        img_file.save(save_img_path)
        try:
            crop_model(save_img_path)

            crop_path = './runs/detect/exp/crops'
            image_class_list = os.listdir(crop_path)
            print(image_class_list)

            # print('C:/Users/HP/Desktop/LastProject/last/runs/detect/exp/crops/*/*.jpg')
            #
            # crop_path = 'C:/Users/HP/Desktop/LastProject/last/runs/detect/exp/crops'
            # for i,furnitures in enumerate(glob(crop_path + '/*/*.jpg')):


            crop_img_path = glob('./runs/detect/exp/crops/*/*.jpg')

            # print(crop_img_path)

            for name in crop_img_path:
                name = name.replace("\\","/")
                src = os.path.join(crop_path,name.split("/")[-2] +'/'+ name.split("/")[-1]).replace("\\","/")
                # print(src)
                dst = name.split("/")[-2] + '_' + name.split("/")[-1]
                dst = os.path.join(crop_path,name.split("/")[-2] +'/'+ dst).replace("\\","/")

                os.rename(src, dst)

            # aws region
            location = 'ap-northeast-2'

            aws_images_url = {}
            for imgs in glob('./runs/detect/exp/crops/*/*.jpg'):
                # print(imgs)
                imgs = imgs.replace("\\","/")
                img_file_name = imgs.split('/')[-2] + imgs.split('/')[-1]
                object_name = imgs.split('/')[-1]
                handle_upload_img(imgs, object_name)
                image_url = f'https://{BUCKET_NAME}.s3.{location}.amazonaws.com/{object_name}'
                aws_images_url[f'{object_name}'] = image_url

            if os.path.exists('Detect_image_folder'):
                os.remove(save_img_path)

            if os.path.exists('./runs/detect/exp/crops'):
                shutil.rmtree('./runs')

        except:
            shutil.rmtree('./runs')


        return aws_images_url

    # 가구 누를 시 net 통해서 url 받고 그 url 사진 벡터전환 후 이미지 유사도 검색
    @app.route('/click_image', methods=['POST'])
    def Click_Image():
        data = request.get_json()
        data_id = data['id']
        data_url = data['url']

        savelocation = f"./Embedding_image_folder/{data_id}.jpg"
        urllib.request.urlretrieve(data_url, savelocation)
        print("저장되었습니다!")

        sim_list = image_to_vec(savelocation)
        print(sim_list)

        if os.path.exists('Embedding_image_folder'):
            os.remove(savelocation)

        return sim_list

    # 가구 누를 시 이미지 유사도 계산
    @app.route('/get_image',methods=['POST'])
    def Get_Image_Embedding():
        img_file = request.files['img']
        print(img_file)

        save_img_path = f'Embedding_image_folder/{img_file.filename}'
        img_file.save(save_img_path)
        # print('bbbbbbbbbbb')

        sim_list = image_to_vec(save_img_path)
        print(sim_list)

        if os.path.exists('Embedding_image_folder'):
            os.remove(save_img_path)

        # ## 바이트 가져오기
        # print(img_file.read())

        return sim_list
    
    # url통해서 이미지 저장하고 vector변환 후 csv 저장
    @app.route('/urlimg_to_vec', methods=['POST'])
    def Test_images():
        # {"id" : "ID값", "url" : "이미지 경로"}
        data = request.get_json()
        print(data)
        data_id = data['id']
        # print(data_id)
        data_url = data['url']
        # print(data_url)

        # print(type(data))
        # print(data_url)

        savelocation = f"./Get_net_image_folder/{data_id}.jpg"
        urllib.request.urlretrieve(data_url, savelocation)
        print("저장되었습니다!")

        img_vec_df = pd.DataFrame(get_vector(savelocation)).T
        img_vec_df['image'] = savelocation.split('/')[-1]

        img_vec_df.rename(columns=lambda x: str(x), inplace=True)

        with open('Image_similarity/Embedding_img.csv', mode='a', newline='') as f:
            csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            csv_writer.writerow(img_vec_df.iloc[0])
            f.close()

        print("벡터 변환 완료하였습니다!")
        os.remove(savelocation)

        return "success"

    return app

if __name__ == "__main__":
    app = create_app()
    app.run(host='0.0.0.0', port=5000)