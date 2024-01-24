import os
import shutil

def collect_files(source_dir, destination_dir, excluded_folder):
    # source_dir 내의 모든 파일과 폴더 목록을 얻음
    for root, dirs, files in os.walk(source_dir):
        # 만약 excluded_folder가 dirs 리스트에 있다면 해당 폴더는 건너뜀
        if excluded_folder in dirs:
            dirs.remove(excluded_folder)
            continue
        
        for file in files:
            file_path = os.path.join(root, file)  # 파일 경로 생성
            # 파일을 destination_dir로 복사
            shutil.copy(file_path, destination_dir)

# 폴더 내 파일을 모을 source 디렉토리와 파일들을 모을 destination 디렉토리를 설정
source_directory = 'C:/Users/user/Desktop/rasberrypi_video_save/frame/'  # 여기에 복사하고 싶은 폴더의 경로를 입력하세요
destination_directory = 'C:/Users/user/Desktop/rasberrypi_video_save/merge/'  # 여기에 파일들을 모을 폴더의 경로를 입력하세요
excluded_folder_name = '_'  # 제외할 폴더명 입력

# 파일들을 복사
collect_files(source_directory, destination_directory, excluded_folder_name)


