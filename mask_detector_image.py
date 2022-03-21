# 필요한 패키지 import
# tensorflow : 구글에서 만든 머신러닝 및 딥러닝 프로그램을 쉽게 구현할 수 있도록 다양한 기능을 제공해 주는 라이브러리
# keras : 오픈 소스 신경망 라이브러리(텐서플로우 2.0.0 이상의 버전을 설치했을 경우 텐서플로우에 keras가 내장)
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input # mobilenet_v2 모델에 필요한 형식에 이미지를 적절하게 맞추기위한 함수(전처리)
from tensorflow.keras.preprocessing.image import img_to_array # 이미지를 numpy 배열로 변환
from tensorflow.keras.models import load_model # 모델 로드
import argparse # 명령행 파싱(인자를 입력 받고 파싱, 예외처리 등) 모듈
import numpy as np # 파이썬 행렬 수식 및 수치 계산 처리 모듈
import cv2 # opencv 모듈
import os # 운영체제 기능 모듈

# 실행을 할 때 인자값 추가
ap = argparse.ArgumentParser() # 인자값을 받을 인스턴스 생성
# 입력받을 인자값 등록
ap.add_argument("-i", "--input", required=True, help="input 이미지 경로")
ap.add_argument("-o", "--output", type=str, help="output 이미지 경로") # 이미지 저장 경로
# 입력받은 인자값을 args에 저장
args = vars(ap.parse_args())

# 얼굴 인식 모델 로드
print("[얼굴 인식 모델 로딩]")
face_detector = "./face_detector/"
prototxt = face_detector + "deploy.prototxt"
weights = face_detector + "res10_300x300_ssd_iter_140000.caffemodel"
network = cv2.dnn.readNet(prototxt, weights) # cv2.dnn.readNet() : 네트워크를 메모리에 로드

# Mask Detector 모델 로드
print("[Mask Detector 모델 로딩]")
mask_detector_model = "mask_detector.h5"
model = load_model(mask_detector_model) # load_model() : 모델 로드

# input 이미지 읽기
image = cv2.imread(args["input"])

# 이미지 크기
(H, W) = image.shape[:2]

# blob 이미지 생성
# 파라미터
# 1) image : 사용할 이미지
# 2) scalefactor : 이미지 크기 비율 지정
# 3) size : Convolutional Neural Network에서 사용할 이미지 크기를 지정
blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300), (104.0, 177.0, 123.0))

# 얼굴 인식
print("[얼굴 인식]")
network.setInput(blob) # setInput() : blob 이미지를 네트워크의 입력으로 설정
detections = network.forward() # forward() : 네트워크 실행(얼굴 인식)

# 인식할 최소 확률
minimum_confidence = 0.5

# 얼굴 인식을 위한 반복
for i in range(0, detections.shape[2]):
    # 얼굴 인식 확률 추출
    confidence = detections[0, 0, i, 2]
    
    # 얼굴 인식 확률이 최소 확률보다 큰 경우
    if confidence > minimum_confidence:
        # bounding box 위치 계산
        box = detections[0, 0, i, 3:7] * np.array([W, H, W, H])
        (startX, startY, endX, endY) = box.astype("int")
        
        # bounding box 가 전체 좌표 내에 있는지 확인
        (startX, startY) = (max(0, startX), max(0, startY))
        (endX, endY) = (min(W - 1, endX), min(H - 1, endY))        
        
        # 인식된 얼굴 추출 후 전처리
        face = image[startY:endY, startX:endX] # 인식된 얼굴 추출
        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB) # RGB 변환
        face = cv2.resize(face, (224, 224)) # 이미지(얼굴) 크기 조정
        face = img_to_array(face) # 이미지(얼굴)를 numpy 배열로 변환
        face = preprocess_input(face) # 모델에 필요한 형식에 이미지(얼굴)를 적절하게 맞추기위한 함수(전처리)
        face = np.expand_dims(face, axis=0) # expand_dims() : 차원 확장(axis : 계산할 기준(0 : 열, 1 : 행))
        
        # Mask Detector 수행
        # Model.predict() : 테스트 입력 dataset 에 대한 모델의 출력값 확인
        # mask : 마스크 착용 확률
        # withoutMask : 마스크 미착용 확률
        (mask, without_mask) = model.predict(face)[0]
        
        # bounding box 레이블 설정
        label = "Mask" if mask > without_mask else "No Mask"
        
        # bounding box 색상 설정
        if label == "Mask" and max(mask, without_mask) * 100 >= 70:
            color = (0, 255, 0) # 초록
        elif label == "No Mask" and max(mask, without_mask) * 100 >= 70:
            color = (0, 0, 255) # 빨강
        else:
            color = (0, 255, 255) # 노랑
        
        # 확률 설정
        label = "{}: {:.2f}%".format(label, max(mask, without_mask) * 100)

        # bounding box 출력
        cv2.putText(image, label, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
        cv2.rectangle(image, (startX, startY), (endX, endY), color, 2)

# 이미지 저장
if args["output"] !=  None: # output 이미지 경로를 입력하였을 때(입력하지 않은 경우 저장되지 않음)
    cv2.imwrite(args["output"], image) # 파일로 저장, 포맷은 확장자에 따름

# 이미지 show
cv2.imshow("Mask Detector", image)
cv2.waitKey(0)
