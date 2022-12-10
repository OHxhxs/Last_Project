from PIL import Image
import io

import torch

model_yolo = torch.hub.load('yolov5', 'custom',
                                    path='./Detect_Furniture/best.pt',
                                    source='local')




def crop_model(image_path):
    result = model_yolo(image_path)
    print(result.pandas().xyxy[0])
    result.crop()

    return "success"







# def image_to_byte_array(image_path):
#     img = Image.open(image_path)
#     imgByteArr = io.BytesIO()
#     img.save(imgByteArr, format=img.format)
#     imgByteArr = imgByteArr.getvalue()
#
#     return imgByteArr
#
# k = image_to_byte_array('C:/Users/HP/Desktop/LastProject/last/Embedding_image_folder/오현승.jpg')
# print(k)

