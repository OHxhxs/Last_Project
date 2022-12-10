from fastapi import FastAPI, File, UploadFile, Form
from pydantic import BaseModel
from typing import Optional
import uvicorn

import torch
import clip


# 채팅 필터링 모델
from transformers import TextClassificationPipeline, BertForSequenceClassification, AutoTokenizer

import numpy as np
import faiss
from PIL import Image
import io
import urllib.request

import os

# from pyngrok import ngrok
# import nest_asyncio

class fur_upload(BaseModel):
    id: str
    url: str

class fur_del(BaseModel):
    id: str

device = "cuda" if torch.cuda.is_available() else "cpu"
vec_model, preprocess = clip.load("ViT-B/32", device=device)
print(device)

def clip_get_vector(image_path):
    image = preprocess(image_path).unsqueeze(0).to(device)

    # 이미지 임배딩 벡터값(특징 추출)
    with torch.no_grad():
        image_features = vec_model.encode_image(image)

    return image_features[0]



# 비속어 필터링 모델
model_name = 'smilegate-ai/kor_unsmile'

model = BertForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)


# 가구 찾는 모델
model_yolo = torch.hub.load('C:/Users/HP/Desktop/LastProject/last/app/yolov5', 'custom',
                                path='C:/Users/HP/Desktop/LastProject/last/app/Detect_Furniture/best.pt',
                                source='local')

def filter_chatting(text):
    pipe = TextClassificationPipeline(
        model=model,
        tokenizer=tokenizer,
        device=0,  # cpu: -1, gpu: gpu number
        return_all_scores=True,
        function_to_apply='sigmoid'
    )

    filter_text = pipe(text)[0]

    for filter in filter_text[:-1]:
        if filter['score'] > 0.6:
            return filter['label'] + "에 대한 말이 담겨있습니다. 다시 작성해주세요"

    return text


app = FastAPI()

@app.get("/")
async def index():
    return {"message" : "Hello World"}

@app.get("/filter/{text}")
async def Filtering(text:str):
    filter_text = filter_chatting(text)
    return {"filter_text":filter_text}

@app.post('/upload_furniture')
# async def Upload_Furniture(data : list = Form()):
async def Upload_Furniture(data : fur_upload):
    # data = eval(data[0])
    data = data.dict()
    print(data)
    #
    id = data['id']
    img_url = data['url']

    if os.path.exists("seller_img_folder"):
        pass
    else:
        os.mkdir("seller_img_folder")

    savelocation = f"./seller_img_folder/{id}.jpg"
    urllib.request.urlretrieve(img_url, savelocation)
    print("저장되었습니다!")


    # 벡터 만들기

    img = Image.open(savelocation)
    new_img_vec = clip_get_vector(img).cpu().numpy().reshape(1,-1).astype('float32')

    id = np.array([id]).astype('int64')
    faiss_db_name = "furniture_embedding"

    if os.path.exists(faiss_db_name):
        pass
    else:
        index = faiss.IndexFlatL2(512)
        index = faiss.IndexIDMap2(index)
        print(index.ntotal)
        faiss.write_index(index, faiss_db_name)

    index = faiss.read_index(faiss_db_name)
    index.add_with_ids(new_img_vec, id)

    print("db 내부 총 개수 : ",index.ntotal)

    faiss.write_index(index, faiss_db_name)

    os.remove(savelocation)

    # return {f"{id}": new_img_vec}
    return "Success"

@app.post('/delete_furniture')
async def Upload_Furniture(data : fur_del):
    data = data.dict()
    print(data)

    id = data['id']

    id = np.array([id]).astype('int64')

    faiss_db_name = "furniture_embedding"
    index = faiss.read_index(faiss_db_name)
    print("DB 지우기 전 총 개수 : ",index.ntotal)

    index.remove_ids(id)

    faiss.write_index(index, faiss_db_name)
    print("DB 지운 후 총 개수 : ", index.ntotal)

    return "success"

@app.post('/furniture_detect')
async def Furniture_Detect_Image(file: UploadFile):
    img = await file.read()
    img = Image.open(io.BytesIO(img))
    # print(img)

    result = model_yolo(img)
    df = result.pandas().xyxy[0]
    print(df)

    return_dict = {}
    data_list = []
    for i in range(len(df)):
        # print(df.iloc[i,[0,1,2,3,6]])

        xmin = df.iloc[i,0]
        xmax = df.iloc[i,2]
        ymin = df.iloc[i,1]
        ymax = df.iloc[i,3]


        x = xmin
        y = img.size[1] - ymax
        w = xmax - xmin
        h = ymax - ymin

        class_name = df.iloc[i,6]

        crop_list = {}
        crop_list['crop'] = [x,y,w,h]
        crop_list['category'] = class_name

        data_list.append(crop_list)

    return_dict['datas'] = data_list

    return return_dict

@app.post('/find_sim_image')
async def Find_Similar_Image(file: UploadFile):
    img = await file.read()
    img = Image.open(io.BytesIO(img))

    # query vector, gpu환경이면 .cpu() 빼고 실행
    query_vector = clip_get_vector(img).reshape(1,-1).cpu().numpy().astype('float32')
    # print(query_vector.shape)
    # print(eval(vectors[0])['vec'])

    # # db에 있는 벡터들 다 가져와서 유사도 구하기
    # db_id_list = []
    # db_vec_list = []
    #
    # for id_vec_dict in eval(vectors[0])['vec']:
    #     # print(id_vec_dict)
    #     # id값
    #     db_id_list.append(id_vec_dict['id'])
    #     # vector값들
    #     db_vec_list.append(id_vec_dict['vec'])
    #
    # db_id_list = np.array(db_id_list).astype('int64')
    # db_vec_list = np.array(db_vec_list).astype('float32')

    # print("Total vector shape")
    # print(db_vec_list.shape)
    # print(type(db_id_list))
    # print("id shape")
    # print(db_id_list.shape)

    # index = faiss.IndexFlatL2(db_vec_list.shape[1])
    # index = faiss.IndexIDMap2(index)
    # index.add_with_ids(db_vec_list, db_id_list)

    index = faiss.read_index("furniture_embedding")
    distances, indices = index.search(query_vector, 2)
    print(distances)
    print(indices)
    # print(type(indices))
    return {"id" : indices[0].tolist()}
    # return "success"

if __name__ == "__main__":
    # ngrok_tunnel = ngrok.connect(5000)
    # print("Public URL : ", ngrok_tunnel.public_url)
    # nest_asyncio.apply()
    # uvicorn.run(app, port=5000)
    uvicorn.run(app, host='192.168.0.42',port=5000)