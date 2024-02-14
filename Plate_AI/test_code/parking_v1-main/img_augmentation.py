##ile_path = 'C:/Users/user/Desktop/__/'


import random
import numpy as np
import os
import cv2
import glob
from PIL import Image
import PIL.ImageOps    




#다음 변수를 수정하여 새로 만들 이미지 갯수를 정합니다.
num_augmented_images = 50

file_path = 'C:/Users/user/Desktop/새 폴더/'
file_names = os.listdir(file_path)
print('file_names', file_names)

total_origin_image_num = len(file_names)


def imread(filename, flags=cv2.IMREAD_COLOR, dtype=np.uint8):
    try:
        n = np.fromfile(filename, dtype)
        img = cv2.imdecode(n, flags)
        return img
    except Exception as e:
        print(e)
        return None
    
    
def imwrite(filename, img, params=None):
    try:
        ext = os.path.splitext(filename)[1]
        result, n = cv2.imencode(ext, img, params)

        if result:
            with open(filename, mode='w+b') as f:
                n.tofile(f)
            return True
        else:
            return False
    except Exception as e:
        print(e)
        return False
    

for i in os.listdir(file_path):
    
    img_path = file_path + i
    print('img_path', img_path)
    
    base_name = os.path.basename(img_path)
    base_name, _ = base_name.split(".")
    print('base_name', base_name)
            
    image = Image.open(img_path)
    random_augment = random.randrange(1,5)
    
    if(random_augment == 1):
        #이미지 기울이기
        print("rotate")
        rotated_image = image.rotate(random.randrange(-8, 8))
        rotated_image.save(file_path + 'rotated_' + base_name + '.png')
        
    elif(random_augment == 2):
        #노이즈 추가하기
        print(img_path)
        img = imread(img_path)
        print("noise")
        row, col, ch = img.shape
        mean = 0
        sigma = 3
        gauss = np.random.normal(mean, sigma, (row, col, ch))
        noisy_image = np.clip(image + gauss, 0, 255)
        imwrite(file_path + 'noized_' + base_name + '.png', noisy_image)
        
    elif(random_augment == 3):
        img = imread(img_path)
        print("resize")
               
        # 랜덤한 비율로 축소 또는 확대를 위한 스케일 팩터 생성
        scale_factor = np.random.uniform(0.2, 2.0)

        # 이미지 크기 가져오기
        height, width = img.shape[:2]

        # 새로운 크기 계산
        new_height = int(height * scale_factor)
        new_width = int(width * scale_factor)

        # 이미지 크기 조정
        resized_image = cv2.resize(img, (new_width, new_height))
        
        imwrite(file_path + 'resized_' + base_name + '.png', resized_image)
        
    else:        
        img = imread(img_path)
        
        # 랜덤한 비율로 축소 또는 확대를 위한 스케일 팩터 생성
        scale_factor = np.random.uniform(0.2, 2.0)
        height, width = img.shape[:2]
        
        # 새로운 크기 계산
        new_height = int(height * scale_factor)
        new_width = int(width * scale_factor)

        # 이미지 크기 조정
        resized_image = cv2.resize(img, (new_width, new_height))
            
        rotation_angle = random.randrange(-10, 10)
        center = tuple(np.array(resized_image.shape[1::-1]) / 2)
        # 회전 매트릭스 생성
        rotation_matrix = cv2.getRotationMatrix2D(center, rotation_angle, 1.0)
        # 이미지 회전
        mixed_img = cv2.warpAffine(resized_image, rotation_matrix, resized_image.shape[1::-1], flags=cv2.INTER_LINEAR)  
        
        row, col, ch = mixed_img.shape
        mean = 0
        sigma = 10
        gauss = np.random.normal(mean, sigma, (row, col, ch))

        # gauss 배열을 mixed_img 배열과 동일한 모양으로 조정
        gauss = cv2.resize(gauss, (col, row))

        noisy_image = np.clip(mixed_img + gauss, 0, 255)
        imwrite(file_path + 'mixed_' + base_name + '.png', noisy_image)




        
