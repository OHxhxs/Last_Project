import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torch.autograd import Variable

from sklearn.metrics.pairwise import cosine_distances,pairwise_distances,cosine_similarity

from PIL import Image
import pandas as pd

import os

print(os.getcwd())
# print(os.listdir())
# Load the pretrained model
model = torch.load('./Image_similarity/furniture_resnet34.pth', map_location=torch.device('cpu'))

# Use the model object to select the desired layer
layer = model._modules.get('avgpool')

model.eval()

scaler =  transforms.Resize((224,224))
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
to_tensor = transforms.ToTensor()

def get_vector(image_name):
    # 1. Load the image with Pillow library
    img = Image.open(image_name).convert('RGB')
    # 2. Create a PyTorch Variable with the transformed image
    t_img = Variable(normalize(to_tensor(scaler(img))).unsqueeze(0))
    # print(t_img[:,:,:].shape)
    # print(t_img)
    # 3. Create a vector of zeros that will hold our feature vector
    #    The 'avgpool' layer has an output size of 512
    my_embedding = torch.zeros(512)

    # print(my_embedding)
    # 4. Define a function that will copy the output of a layer
    def copy_data(m, i, o):
        my_embedding.copy_(o.data.reshape(o.data.size(1)))

    # 5. Attach that function to our selected layer
    h = layer.register_forward_hook(copy_data)
    # 6. Run the model on our transformed image
    model(t_img)
    # 7. Detach our copy function from the layer
    h.remove()
    # 8. Return the feature vector
    return my_embedding

# def fetch_most_similar_products(image_name,n_similar=7):
#     # print("-----------------------------------------------------------------------")
#     # print("Original Product:")
#     # show_img(image_name,image_name)
#     curr_index = embedding_df[embedding_df['image']==image_name].index[0]
#     closest_image = pd.DataFrame(cosine_similarity_df.iloc[curr_index].nlargest(n_similar+1)[1:])
#     print("-----------------------------------------------------------------------")
#     print("Recommended Product")
#     for index,imgs in closest_image.iterrows():
#         similar_image_name = embedding_df.iloc[index]['image']
#         similarity = np.round(imgs.iloc[0],3)
#         # show_img(similar_image_name,str(similar_image_name)+' nSimilarity : '+str(similarity))

def image_to_vec(image_path):

    # 이미지를 벡터로 만들고 데이터프레임화
    img_vec_df = pd.DataFrame(get_vector(image_path)).T
    # 새로운 컬럼 image생성
    img_vec_df['image'] = image_path.split('/')[-1]
    img_vec_df.rename(columns=lambda x: str(x), inplace=True)
    # print('1'*50)
    # print(img_vec_df.head())
    # 전에 만들어 놨던 이미지들의 벡터 데이터프레임
    embedding_df = pd.read_csv("./Image_similarity/Embedding_img.csv")
    # del embedding_df['Unnamed: 0']

    # print('2' * 50)
    # print(embedding_df.head())
    # 새로 들어온 이미지와 전에 있던 이미지들의 벡터 데이터프레임 합치기
    get_img_embedding_df = pd.concat([embedding_df, img_vec_df])
    get_img_embedding_df.reset_index(inplace=True, drop=True)

    # print(get_img_embedding_df[get_img_embedding_df.duplicated()])

    # print('3' * 50)
    # print(get_img_embedding_df.tail())
    # 코사인 유사도 계산 후 데이터프레임화
    cosine_similarity_df = pd.DataFrame(cosine_similarity(get_img_embedding_df.drop('image', axis=1)))

    # print('4' * 50)
    # print(cosine_similarity_df.head())

    # 들어온 이미지의 이름이 같은거와 유사한 10개의 index추출
    curr_index = get_img_embedding_df[get_img_embedding_df['image'] == image_path.split('/')[-1]].index[0]
    print(curr_index)
    # 그 인덱스로 유사한 이미지들 추출
    closest_image = pd.DataFrame(cosine_similarity_df.iloc[curr_index].nlargest(6)[1:])
    print(closest_image)

    sim_list = []
    for i in closest_image.index:
        sim_list.append(get_img_embedding_df.iloc[i-1,-1])
        # print(get_img_embedding_df.iloc[i-1,-1])

    return sim_list

# if __name__=='__main__':



