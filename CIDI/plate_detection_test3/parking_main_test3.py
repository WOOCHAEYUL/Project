import time
import ai_test3
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
    formatter = logging.Formatter('%(asctime)s - %(message)s')
    file_handler.setFormatter(formatter)
    
    # 기존 핸들러 제거 후 새로운 핸들러 추가
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

def process_image(image_path, result_folder):
    return ai_test3.state(image_path, result_folder)

def state_machine():
    if mode_state == "A0104":
        logging.info(f"@@@@@@@@@@@@ 물체 접근@@@@@@@@@@@ ::: mode_state =>  {mode_state}")
        
        img_base_folder = '../data_test/'
        result_base_folder = './result/'

        total_tp_all = 0
        total_fp_all = 0        
        total_fn_all = 0

        result_folder = get_next_test_folder(result_base_folder)
        os.makedirs(result_folder, exist_ok=True)
        
        setup_logging_for_folder(result_folder)  # 폴더별로 로깅 설정

        for folder in os.listdir(img_base_folder):
            folder_path = os.path.join(img_base_folder, folder)
            if os.path.isdir(folder_path):
                img_file_list = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
                
                dataset_result_folder = os.path.join(result_folder, folder)
                os.makedirs(dataset_result_folder, exist_ok=True)
                
                total_tp = 0
                total_fp = 0
                total_fn = 0

                for source_path in img_file_list:
                    img = cv2.imread(source_path)
                    if img is None:
                        logging.info("Failed to load image for ROI selection.")
                        continue

                    cv2.namedWindow("Image")
                    cv2.setMouseCallback("Image", ai_test3.click_event)
                    
                    screen_height, screen_width = 1080, 1920
                    img_height, img_width = img.shape[:2]
                    ai_test3.scale_factor = min(screen_width / img_width, screen_height / img_height, 1)
                    resized_img = cv2.resize(img, (int(img_width * ai_test3.scale_factor), int(img_height * ai_test3.scale_factor)))
                    
                    while True:
                        display_img = resized_img.copy()
                        if ai_test3.roi_pts:
                            scaled_pts = [(int(x * ai_test3.scale_factor), int(y * ai_test3.scale_factor)) for x, y in ai_test3.roi_pts]
                            cv2.polylines(display_img, [np.array(scaled_pts, dtype=np.int32)], isClosed=True, color=(0, 255, 0), thickness=2)
                        
                        cv2.imshow("Image", display_img)
                        
                        key = cv2.waitKey(1) & 0xFF
                        if key == ord('r'):
                            ai_test3.roi_pts = []
                            ai_test3.roi_set = False
                        elif key == 13:
                            if ai_test3.roi_set:
                                break
                        elif key == 27:
                            break
                    
                    cv2.destroyAllWindows()
                    
                    tp, fp, fn = process_image(source_path, dataset_result_folder)
                    total_tp += tp
                    total_fp += fp
                    total_fn += fn

                total_tp_all += total_tp
                total_fp_all += total_fp
                total_fn_all += total_fn

                if total_tp + total_fn + total_fp > 0:
                    average_accuracy = total_tp / (total_tp + total_fn + total_fp)
                else:
                    average_accuracy = 0
                
                logging.info('')
                logging.info("********************************************************************")
                logging.info(f"세트: {folder}")
                logging.info(f"Plates - Total GT: {total_tp + total_fn + total_fp}")
                logging.info(f"Total TP(정상인식): {total_tp}, Total FP(오검출): {total_fp}, Total FN(미검출): {total_fn}")
                logging.info(f"Average Accuracy(평균 정확도): {average_accuracy * 100:.2f}%")
                logging.info("********************************************************************")
                logging.info('')

        # 전체 테스트 폴더에 plate_result.xlsx 저장
        ai_test3.save_overall_results(result_folder)

        logging.info("====================================================================")
        logging.info("총 결과")
        logging.info(f"Plates - Total GT: {total_tp_all + total_fn_all + total_fp_all}")
        logging.info(f"Total TP(정상인식): {total_tp_all}, Total FP(오검출): {total_fp_all}, Total FN(미검출): {total_fn_all}")
        logging.info(f"Average Accuracy(평균 정확도): {(total_tp_all / (total_tp_all + total_fn_all + total_fp_all)) * 100:.2f}%")
        logging.info("====================================================================")

if __name__ == '__main__':
    logging.info("[main] Start")
    
    state_result = state_machine()
    logging.info(f"state result =======> {state_result}")
