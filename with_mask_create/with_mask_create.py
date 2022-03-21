# 필요한 패키지 import
import cv2 # opencv 모듈
import os # 운영체제 기능 모듈
from face_masker import create_mask # facemasker.py에 create_mask import

# 마스크를 착용시킬 이미지 경로(마스크를 착용하지 않은 얼굴 이미지)
without_mask_path = "../dataset/without_mask"

# os.path.join('디렉터리 A', '디렉터리 B') = 'A/B' : 경로를 병합하여 새로운 경로 생성
# os.listdir('디렉터리') : 디렉터리 내의 파일 및 서브 디렉터리 리스트
# os.path.isfile() : 파일 경로가 존재하는지 확인
# without_mask_path 경로의 있는 파일(이미지) 리스트
images = [os.path.join(without_mask_path, f) for f in os.listdir(without_mask_path) if os.path.isfile(os.path.join(without_mask_path, f))]

# 마스크 이미지 경로
white_mask_image = "./mask_images/white-mask.png" # 흰색 마스크
black_mask_image = "./mask_images/black-mask.png" # 검 마스크
blue_mask_image = "./mask_images/blue-mask.png" # 파란색 마스크

# 이미지 수만큼 반복
for i in range(len(images)):
    #print(i+1, "번째 이미지 경로 :", images[i])
    if i < 600 : # 600개 이미지 흰색 마스크 적용
        create_mask(images[i], white_mask_image)
    elif i < 900 : # 300개 이미지 검색 마스크 적용
        create_mask(images[i], black_mask_image)
    else : # 나머지 100개 이미지 파란 마스크 적용
        create_mask(images[i], blue_mask_image)
