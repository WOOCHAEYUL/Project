import os
import cv2
from ultralytics import YOLO
from PIL import ImageFont, ImageDraw, Image


find_plate_model = YOLO("model/find_plate_240123.pt")


def detect_plate_v8(img_path) :
    frame = cv2.imread(img_path, cv2.IMREAD_COLOR)
    results = find_plate_model(frame)  # inference
      
    base_name = os.path.basename(img_path)
    base_name = base_name.rstrip('.jpg')
    #print('base_name = {}'.format(base_name))

    obj_list = []

    for result in results:
        for box in result.boxes.data :
            x1, y1, x2, y2, conf, cls = int(box[0]), int(box[1]), int(box[2]), int(box[3]), float(box[4]),  int(box[5])
            if conf >= 0.25: # 0.5 이상으로 확신하는 경우만 바운딩 박스를 그림
                obj_list.append([conf, x1, y1, x2, y2])

    if obj_list :
        obj_list.sort(key= lambda x:x[0])
        x = obj_list[-1][1]
        y = obj_list[-1][2]
        h = obj_list[-1][4] - obj_list[-1][2]
        w = obj_list[-1][3] - obj_list[-1][1]
        crop_img = frame[y : y + h, x : x + w]
        
         
        cv2.imwrite('./crop_img/{}.jpg'.format(base_name), crop_img)
        #cv2.imshow('crop', crop_img)
        #cv2.waitKey(0)
                
        return crop_img, x1, y1, x2, y2        
        
    else :
        print("list empty!")
        return None             


if __name__ == '__main__' :
    print("start LPR Crop")
    img_folder = 'E:/1.Project/1.Thub/3.AISolution/1.scr/ultralytics-main/ultralytics-main/dataset/find_plate/train/images/'
    img_file_list = os.listdir(img_folder)

    for file in img_file_list:
        img_file = img_folder + file
        detect_plate_v8(img_file)

    