import time
import ai_test1
import cv2
import numpy as np
import os
import logging

# 기본 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

mode_state = "A0104"

def setup_logging_for_folder(folder_path):
    log_file_path = os.path.join(folder_path, "plate_detection_log.txt")
    file_handler = logging.FileHandler(log_file_path)
    file_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(logging.StreamHandler())

def get_next_test_folder(base_folder):
    """
    Get the next test folder name like test1, test2, etc.
    """
    test_folders = [d for d in os.listdir(base_folder) if os.path.isdir(os.path.join(base_folder, d)) and d.startswith('test')]
    test_numbers = [int(d.replace('test', '')) for d in test_folders if d.replace('test', '').isdigit()]
    next_number = max(test_numbers, default=0) + 1
    return os.path.join(base_folder, f'test{next_number}')


def load_illegal_plates(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return [line.strip() for line in file.readlines()]

def process_image(image_path, result_folder, illegal_plates):
    return ai_test1.state(image_path, result_folder, illegal_plates)
              
def state_machine():
    if mode_state == "A0104":
        print(f"@@@@@@@@@@@@ 물체 접근@@@@@@@@@@@ ::: mode_state =>  {mode_state}")
        print('')
        
        illegal_plates_file = './disabled_license_plate_info.txt'
        
        if not os.path.exists(illegal_plates_file):
            print(f"File not found: {illegal_plates_file}")
            return
        
        illegal_plates = load_illegal_plates(illegal_plates_file)
        
        img_base_folder = '../data_test/'
        result_base_folder = './result/'

        total_tp_illegal_all = 0
        total_fp_illegal_all = 0        
        total_fn_illegal_all = 0
        total_tp_disabled_all = 0
        total_fp_disabled_all = 0
        total_fn_disabled_all = 0
        
        result_folder = get_next_test_folder(result_base_folder)
        os.makedirs(result_folder, exist_ok=True)
        
        setup_logging_for_folder(result_folder)  # 폴더별로 로깅 설정

        for folder in os.listdir(img_base_folder):
            folder_path = os.path.join(img_base_folder, folder)
            if os.path.isdir(folder_path):
                img_file_list = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
                
                dataset_result_folder = os.path.join(result_folder, folder)
                os.makedirs(dataset_result_folder, exist_ok=True)
                
                total_tp_illegal = 0
                total_fp_illegal = 0
                total_fn_illegal = 0
                total_tp_disabled = 0
                total_fp_disabled = 0
                total_fn_disabled = 0

                for source_path in img_file_list:
                    img = cv2.imread(source_path)
                    if img is None:
                        print("Failed to load image for ROI selection.")
                        continue

                    cv2.namedWindow("Image")
                    cv2.setMouseCallback("Image", ai_test1.click_event)
                    
                    screen_height, screen_width = 1080, 1920
                    img_height, img_width = img.shape[:2]
                    ai_test1.scale_factor = min(screen_width / img_width, screen_height / img_height, 1)
                    resized_img = cv2.resize(img, (int(img_width * ai_test1.scale_factor), int(img_height * ai_test1.scale_factor)))
                    
                    while True:
                        display_img = resized_img.copy()
                        if ai_test1.roi_pts:
                            scaled_pts = [(int(x * ai_test1.scale_factor), int(y * ai_test1.scale_factor)) for x, y in ai_test1.roi_pts]
                            cv2.polylines(display_img, [np.array(scaled_pts, dtype=np.int32)], isClosed=True, color=(0, 255, 0), thickness=2)
                        
                        cv2.imshow("Image", display_img)
                        
                        key = cv2.waitKey(1) & 0xFF
                        if key == ord('r'):
                            ai_test1.roi_pts = []
                            ai_test1.roi_set = False
                        elif key == 13:
                            if ai_test1.roi_set:
                                break
                        elif key == 27:
                            break
                    
                    cv2.destroyAllWindows()
                    
                    tp_illegal, fp_illegal, fn_illegal, tp_disabled, fp_disabled, fn_disabled = process_image(source_path, dataset_result_folder, illegal_plates)
                    total_tp_illegal += tp_illegal
                    total_fp_illegal += fp_illegal
                    total_fn_illegal += fn_illegal
                    total_tp_disabled += tp_disabled
                    total_fp_disabled += fp_disabled
                    total_fn_disabled += fn_disabled

                total_tp_illegal_all += total_tp_illegal
                total_fp_illegal_all += total_fp_illegal
                total_fn_illegal_all += total_fn_illegal
                total_tp_disabled_all += total_tp_disabled
                total_fp_disabled_all += total_fp_disabled
                total_fn_disabled_all += total_fn_disabled

                if total_tp_illegal + total_fn_illegal + total_fp_illegal > 0:
                    average_accuracy_illegal = total_tp_illegal / (total_tp_illegal + total_fn_illegal + total_fp_illegal)
                else:
                    average_accuracy_illegal = 0
                
                if total_tp_disabled + total_fn_disabled + total_fp_disabled > 0:
                    average_accuracy_disabled = total_tp_disabled / (total_tp_disabled + total_fn_disabled + total_fp_disabled)
                else:
                    average_accuracy_disabled = 0
                
                logging.info('')
                logging.info("********************************************************************")
                logging.info(f"세트: {folder}")
                logging.info(f"Illegal Plates - Total GT: {total_tp_illegal + total_fn_illegal + total_fp_illegal}")
                logging.info(f"Total TP(정상인식): {total_tp_illegal}, Total FP(오검출): {total_fp_illegal}, Total FN(미검출): {total_fn_illegal}")
                logging.info(f"Average Accuracy(평균 정확도): {average_accuracy_illegal * 100:.2f}%")
                logging.info(f"Disabled Plates - Total GT: {total_tp_disabled + total_fn_disabled + total_fp_disabled}")
                logging.info(f"Total TP(정상인식): {total_tp_disabled}, Total FP(오검출): {total_fp_disabled}, Total FN(미검출): {total_fn_disabled}")
                logging.info(f"Average Accuracy(평균 정확도): {average_accuracy_disabled * 100:.2f}%")
                logging.info("********************************************************************")
                logging.info('')

        logging.info("====================================================================")
        logging.info("총 결과")
        logging.info(f"Illegal Plates - Total GT: {total_tp_illegal_all + total_fn_illegal_all + total_fp_illegal_all}")
        logging.info(f"Total TP(정상인식): {total_tp_illegal_all}, Total FP(오검출): {total_fp_illegal_all}, Total FN(미검출): {total_fn_illegal_all}")
        logging.info(f"Average Accuracy(평균 정확도): {(total_tp_illegal_all / (total_tp_illegal_all + total_fn_illegal_all + total_fp_illegal_all)) * 100:.2f}%")
        logging.info(f"Disabled Plates - Total GT: {total_tp_disabled_all + total_fn_disabled_all + total_fp_disabled_all}")
        logging.info(f"Total TP(정상인식): {total_tp_disabled_all}, Total FP(오검출): {total_fp_disabled_all}, Total FN(미검출): {total_fn_disabled_all}")
        logging.info(f"Average Accuracy(평균 정확도): {(total_tp_disabled_all / (total_tp_disabled_all + total_fn_disabled_all + total_fp_disabled_all)) * 100:.2f}%")
        logging.info("====================================================================")
        
        ai_test1.save_overall_results(result_folder)

if __name__ == '__main__':
    logging.info("[main] Start")
    
    state_result = state_machine()
    #logging.info(f"state result =======> {state_result}")