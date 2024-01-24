#-*- encoding: utf8 -*-

from ultralytics import YOLO

#yaml_path = './ultralytics/cfg/datasets/defect.yaml'
#yaml_path = 'E:/1.Project/1.Thub/3.AISolution/1.scr/ultralytics-main/ultralytics-main/dataset/sealing_number/sn.yaml'
yaml_path = 'E:/1.Project/1.Thub/3.AISolution/1.scr/ultralytics-main/ultralytics-main/dataset/find_plate/find_plate_v8.yaml' #train19
#yaml_path = 'E:/1.Project/1.Thub/3.AISolution/1.scr/ultralytics-main/ultralytics-main/dataset/voc/voc.yaml' #train20
#yaml_path = 'E:/1.Project/1.Thub/3.AISolution/1.scr/ultralytics-main/ultralytics-main/dataset/plate_char/plate_char.yaml'


if __name__ == '__main__':
    # Load a model
    model = YOLO('yolov8l.yaml')  # build a new model from YAML
    #model = YOLO('yolov8l.pt')  # load a pretrained model (recommended for training)
    model = YOLO('yolov8l.yaml').load('yolov8l.pt')  # build from YAML and transfer weights
    #model = YOLO('E:/1.Project/1.Thub/3.AISolution/1.scr/ultralytics-main/ultralytics-main/runs/detect/train17/weights/last.pt')  # load a pretrained model (recommended for training)
    #model = YOLO('yolov8l.yaml').load('E:/1.Project/1.Thub/3.AISolution/1.scr/ultralytics-main/ultralytics-main/runs/detect/train17/weights/last.pt')  # build from YAML and transfer weights

    # Train the model
    results = model.train(data=yaml_path, epochs=200, imgsz=640)
    