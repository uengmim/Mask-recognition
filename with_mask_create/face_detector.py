# 필요한 패키지 import
import numpy as np # 파이썬 행렬 수식 및 수치 계산 처리 모듈
import cv2 # opencv 모듈
import imutils # 파이썬 OpenCV가 제공하는 기능 중 복잡하고 사용성이 떨어지는 부분을 보완(이미지 또는 비디오 스트림 파일 처리 등)
import os # 운영체제 기능 모듈
import face_recognition # 얼굴 특성 정보 추출(얼굴 인식) 모듈

# 얼굴을 추출할 이미지 경로
face_path = "./face_images"

# os.path.join('디렉터리 A', '디렉터리 B') = 'A/B' : 경로를 병합하여 새로운 경로 생성
# os.listdir('디렉터리') : 디렉터리 내의 파일 및 서브 디렉터리 리스트
# os.path.isfile() : 파일 경로가 존재하는지 확인
# face_path 경로의 있는 파일(이미지) 리스트
images = [os.path.join(face_path, f) for f in os.listdir(face_path) if os.path.isfile(os.path.join(face_path, f))]

# 이미지 수만큼 반복
for i in range(len(images)):
    # 이미지 파일
    image_path = images[i]

    # 이미지 읽기
    face_image = cv2.imread(image_path)

    # face_recognition.face_locations(이미지(numpy 배열), 모델) : 이미지에서 사람 얼굴의 bounding boxes 반환
    face_locations = face_recognition.face_locations(face_image, model='hog') # hog(기본값) : 비교적 덜 정확하지만 cpu에서도 빠름(gpu를 사용 가능한 경우 cnn)
    
    found_face = False # 얼굴이 있는지 없는지 판별

    # 얼굴 인식 목록 수 만큼 반복
    for face_location in face_locations:
        (y1, x2, y2, x1) = face_location # 인식된 얼굴 좌표

        found_face = True # 얼굴이 있음

    if found_face == False:
        print("얼굴을 인식하지 못함 :", image_path)
        continue

    # 이미지 자르기
    face_image = face_image[y1 - 5 : y2 + 5, x1 - 5 : x2 + 5] # [시작 height : 끝 height, 시작 width : 끝 width]
    
    image_path_splits = os.path.splitext(image_path) # os.path.splitext(파일 경로) : 파일의 확장자 추출
    number = image_path_splits[0].split('_') # 이미지 이름 숫자 추출
    without_mask_image_path = '../dataset/without_mask/without_mask_' + number[2] + image_path_splits[1] # 이미지 저장 경로
    
    # 이미지 저장
    cv2.imwrite(without_mask_image_path, face_image) # 파일로 저장, 포맷은 확장자에 따름
    print('저장 경로 :', without_mask_image_path)
