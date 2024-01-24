import cv2
from ultralytics import YOLO
from PIL import ImageFont, ImageDraw, Image
import numpy as np
import csv
import os

# Detection Draw Text
def myPutText(src, text, pos, font_size, font_color) :
    img_pil = Image.fromarray(src)
    draw = ImageDraw.Draw(img_pil)
    font = ImageFont.truetype('fonts/gulim.ttc', font_size)
    draw.text(pos, text, font=font, fill= font_color)
    return np.array(img_pil)

COLOR = (255,0,0) # Font Color(blue)
FONT_SIZE = 30

find_plate_model = YOLO("./model/find_plate_240123.pt")
plate_char_model = YOLO("./model/plate_char_240118.pt")

classes_type = ['plate1', 'plate2']
classes = ['1','2','3','4','5','6','7','8','9','0',
           '가', '나', '다', '라', '마',
           '거', '너', '더', '러', '머', '버', '서', '어', '저',
           '고', '노', '도', '로', '모', '보', '소', '오', '조',
           '구', '누', '두', '루', '무', '부', '수', '우', '주',
           '아', '바', '사', '자', '허', '하', '호', '배', 
           '서울', '서울', '부산', '부산', '대구', '대구', '인천', '인천', 
           '광주', '광주', '대전', '대전', '울산', '울산', '세종', '세종', 
           '경기', '경기', '강원', '강원', '충북', '충북', '충남', '충남', 
           '전북', '전북', '전남', '전남', '경북', '경북', '경남','경남', 
           '제주', '제주', '육', '해', '공', '국', '합', '초', '퍼']


def detect_plate_v8(img_path) :
    frame = cv2.imread(img_path, cv2.IMREAD_COLOR)
    results = find_plate_model(frame)  # plate inference    
    
    base_name = os.path.basename(img_path)
    base_name = base_name.rstrip('.jpg')

    obj_list = []

    for result in results:
        for box in result.boxes.data :
            x1, y1, x2, y2, conf, cls = int(box[0]), int(box[1]), int(box[2]), int(box[3]), float(box[4]),  int(box[5])
            if conf >= 0.5: # 0.5 이상으로 확신하는 경우만 바운딩 박스를 그림
                label = classes_type[cls]
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
                      
        return crop_img, x1, y1, x2, y2, cls        
        
    else :
        print("list empty!")
        return None


def detect_char_v8(frame, plate_type):
    results = plate_char_model(frame)  # plate char inference
    obj_list = []

    for result in results:
        for box in result.boxes.data :
            x1, y1, x2, y2, conf, cls = int(box[0]), int(box[1]), int(box[2]), int(box[3]), float(box[4]),  int(box[5])
            if conf >= 0.2: #0.7 # 0.9 이상으로 확신하는 경우만 바운딩 박스를 그림
                label = classes[cls]
                obj_list.append([label, x1, y1, x2, y2])                
     
            cv2.rectangle(frame, (x1,y1), (x2,y2), (255,0,0), 2)
            cv2.imshow('frame', frame)
            #cv2.waitKey(0)   
           
    if obj_list :
        print('obj_list',obj_list)        
        
        if plate_type == 0:
            # x 값을 기준으로 정렬
            sorted_by_x = sorted(obj_list, key=lambda x: x[1])  
            print("x 값 기준 정렬: ", sorted_by_x)
            
            plate_number = ''
            for i in sorted_by_x:
                plate_number += i[0]

            plate_number= plate_number.replace(" ", "")
            print(plate_number)          
            
        elif plate_type == 1:
        
            top_line = [item for item in obj_list if item[2] < (max(item[2] for item in obj_list) + min(item[2] for item in obj_list)) / 2]
            bottom_line = [item for item in obj_list if item[2] >= (max(item[2] for item in obj_list) + min(item[2] for item in obj_list)) / 2]
            
            print('top_line', top_line)
            print('bottom_line', bottom_line)            
            
            # 각 라인에서 y 좌표를 기준으로 정렬
            top_line.sort(key=lambda x: x[2])
            bottom_line.sort(key=lambda x: x[2])

            # 각 라인에서 x 좌표를 기준으로 정렬
            top_line.sort(key=lambda x: x[1])
            bottom_line.sort(key=lambda x: x[1])

            # 결과 출력
            result = top_line + bottom_line
            print('result::::',result)
            
            plate_number = ''
            for i in result:
                plate_number += i[0]

            plate_number= plate_number.replace(" ", "")
            print("plate_number", plate_number)
            
        return plate_number
    
    else :
        print("list empty!")
        return None
                        
    """
    if obj_list :
        print('obj_list',obj_list)
        
        sorted_by_x = sorted(obj_list, key=lambda x: x[1])  # x 값을 기준으로 정렬
        print("x 값 : ", sorted_by_x)
        
        sorted_by_y = sorted(sorted_by_x, key=lambda x: x[2]) # y 값을 기준으로 정렬
        print("y 값 : ", sorted_by_y)            
        
        diff_index = 0
        for i in range(1, len(sorted_by_y)):
            if abs(sorted_by_y[i][2] - sorted_by_y[i - 1][2]) >= 10: # 높이 차이가 10 이상 나는 인덱스 찾기
                diff_index = i
                break

        # 번호판 위 아래 line 구분
        line_1 = sorted_by_y[:diff_index]
        line_2 = sorted_by_y[diff_index:]
        #print("line_1:", line_1)
        #print("line_2:", line_2)
        
        sorted_line_1 = sorted(line_1, key=lambda x: x[1])  # 두 번째 요소(인덱스 1)를 기준으로 정렬
        #print("line_1 정렬 결과:", sorted_line_1)        
        sorted_line_2 = sorted(line_2, key=lambda x: x[1])  # 두 번째 요소(인덱스 1)를 기준으로 정렬
        #print("line_2 정렬 결과:", sorted_line_2)
        
        combined_lines = sorted_line_1 + sorted_line_2
        print("합쳐진 리스트:", combined_lines)
        
        plate_number = ''
        for i in combined_lines:
            plate_number += i[0]

        plate_number= plate_number.replace(" ", "")
        #print(plate_number)        
            
        return plate_number
    """       
              
def state_machine(img_file):
    global work_mode
    global parking_mode
    global laserData
    global plate_number
      
    img = cv2.imread(img_file)                
    img_copy = img.copy()                
    
    img_name = os.path.basename(img_file)
    print('###########img_name################', img_name)      
    
    cv2.imwrite('./capture_img/{}'.format(img_name), img)
    crop_result = detect_plate_v8('./capture_img/{}'.format(img_name))                    

    if crop_result is not None:
        crop, x1, y1, x2, y2, plate_type = detect_plate_v8('./capture_img/{}'.format(img_name))
        
        #crop, x1, y1, x2, y2 = crop_result
        plate_number = detect_char_v8(crop, plate_type)
        print('plate_number : {}'.format(plate_number))
        
        if plate_number is not None:
            cv2.rectangle(img_copy, (x1,y1), (x2,y2), (255,0,0), 3)      
                      
            base_name = os.path.splitext(img_name)[0] # 이미지 이름 GT 활용
            file_parts = base_name.split("_")
            img_base_name = file_parts[0] if file_parts else base_name
            #print(img_base_name)
            
            # plate 문자 결과값 저장                       
            with open("plate_result.csv", "a", newline='', encoding='utf-8') as csvfile:
                csv_writer = csv.writer(csvfile)
                csv_writer.writerow([img_name, img_base_name, plate_number])                
                
            img_copy = myPutText(img_copy, plate_number, (x1,y1-30), FONT_SIZE, COLOR)
            cv2.imshow('img_copy', img_copy)
            cv2.waitKey(0)

        else:
            print('not char_{}'.format(img_name))
        
    else: 
        print("not plate") 

  
if __name__ == '__main__' :
    print("start LPR Test Program")

    img_folder = 'C:/Users/user/Desktop/merge/'
    img_file_list = os.listdir(img_folder)

    for file in img_file_list:
        img_file = img_folder + file
         
        state_machine(img_file)

    