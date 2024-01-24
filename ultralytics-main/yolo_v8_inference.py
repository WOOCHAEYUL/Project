from ultralytics import YOLO
import os


if __name__ == '__main__':
    # Load a model
    #model = YOLO('./runs/detect/train/weights/last.pt')  
    model = YOLO('E:/1.Project/1.Thub/3.AISolution/1.scr/ultralytics-main/ultralytics-main/runs/detect/train8/weights/best.pt')#('./runs/detect/train23/weights/best.pt')   
    #image_path = "./ultralytics/cfg/data/valid/images/"
    image_path = "C:/Users/user/Desktop/labeling/"#"C:/workspace/Yolo_mark-master/Yolo_mark-master/x64/Release/data/img/"


    for i in os.listdir(image_path):
        image_file = image_path + i
        
        if '.jpg' in image_file:
            print('image_file: ', image_file)
            
   
        # Predict
        results = model.predict(image_file, save=True, save_txt=True)
        res_plot = results[0].plot()

       
