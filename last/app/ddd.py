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

def vector_to_dataframe(image_path):
  image_vector = clip_get_vector(image_path)

  img_vec_df = pd.DataFrame(image_vector.contiguous()).T
  img_vec_df['id'] = image_path.split('/')[-1]

  return img_vec_df

def create_and_search_index(embedding_size, db_embeddings, query_embeddings, k):
	# 특정 embedding size(32)의 faiss index 생성
    index = faiss.IndexFlatL2(embedding_size)
    # db 등록
    index.add(np.ascontiguousarray(db_embeddings))
    # k개의 유사한 값 search
    # I는 db의 index / D는 query와 해당 index의 db와의 distance
    D, I = index.search(query_embeddings, k=k)

    return D, I


def create_app():
    app = Flask(__name__)

    @app.route('/')
    def index():
        return '안녕하세요'

    @app.route('/index')
    def reIndex():
        return "안녕히 가세요"

    # 이미지를 벡터로 변환 후 return
    # data = {"id" : id값 , "url" : "htpps://~" }
    @app.route('/img_to_vec',methods=["POST"])
    def Image_To_Vec():
        data = request.get_json()
        print(data)
        data_id = data['id']
        # print(data_id)
        data_url = data['url']
        # print(data_url)

        # print(type(data))
        # print(data_url)

        savelocation = f"./1_save_url_image/{data_id}.jpg"
        urllib.request.urlretrieve(data_url, savelocation)
        print("저장되었습니다!")

        image_vector = clip_get_vector(savelocation).tolist()
        print(image_vector)

        if os.path.exists('1_save_url_image'):
            os.remove(savelocation)

        return jsonify({data_id : image_vector})

    # 유사한 이미지들의 id return
    @app.route('/find_similarity_image_id', methods=['POST'])
    def Find_Similarity_Image_Id():
        data = request.get_json()
        print("aaa")
        print(data)
        data_url = data['data'][0]['url']

        # url 통해서 이미지 저장
        savelocation = f"./2_save_crop_url_image/{data_url.split('/')[-1]}"
        urllib.request.urlretrieve(data_url, savelocation)
        print("저장되었습니다!")

        # 저장한 이미지 벡터화
        query_img_vectors = clip_get_vector(savelocation)

        # tenor가 gpu이기에 사용 불가
        # query_img_vectors = np.array(query_img_vectors).reshape(1, -1)

        print(query_img_vectors)

        # cpu로 변환해서 사용
        query_img_vectors = query_img_vectors.cpu.numpy().reshape(1, -1)

        data_vec = data['data'][1][0]['vec']
        id_list = []
        vec_list = []
        for id_vec_dict in data_vec:
            # print(id_vec_dict)
            id = list(id_vec_dict.keys())[0]
            vec = id_vec_dict[id]
            id_list.append(id)
            vec_list.append(vec)

        # 벡터의 차원
        d = 512

        print(vec_list)

        # db에 있는 벡터들
        total_vectors = vec_list.cpu().numpy()

        D, I = create_and_search_index(512, total_vectors, query_img_vectors, 2)

        return I

    return app

if __name__ == "__main__":
    app = create_app()
    app.run(host='0.0.0.0', port=5000)
