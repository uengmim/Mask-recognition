# 필요한 패키지 import
# tensorflow : 구글에서 만든 머신러닝 및 딥러닝 프로그램을 쉽게 구현할 수 있도록 다양한 기능을 제공해 주는 라이브러리
# keras : 오픈 소스 신경망 라이브러리(텐서플로우 2.0.0 이상의 버전을 설치했을 경우 텐서플로우에 keras가 내장)
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input # mobilenet_v2 모델에 필요한 형식에 이미지를 적절하게 맞추기위한 함수(전처리)
from tensorflow.keras.preprocessing.image import img_to_array # 이미지를 numpy 배열로 변환
from tensorflow.keras.models import load_model # 모델 로드

from imutils.video import FPS
import imutils # 파이썬 OpenCV가 제공하는 기능 중 복잡하고 사용성이 떨어지는 부분을 보완(이미지 또는 비디오 스트림 파일 처리 등)
import time # 시간 처리 모듈
import argparse # 명령행 파싱(인자를 입력 받고 파싱, 예외처리 등) 모듈
import numpy as np # 파이썬 행렬 수식 및 수치 계산 처리 모듈
import cv2 # opencv 모듈
import os # 운영체제 기능 모듈

# 얼굴 인식 및 Mask Detector 수행 함수
def mask_detector(frame, network, model):
    # 프레임 크기
    (h, w) = frame.shape[:2]
    
    # blob 이미지 생성
    # 파라미터
    # 1) image : 사용할 이미지
    # 2) scalefactor : 이미지 크기 비율 지정
    # 3) size : Convolutional Neural Network에서 사용할 이미지 크기를 지정
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))
    
    # 얼굴 인식
    network.setInput(blob) # setInput() : blob 이미지를 네트워크의 입력으로 설정
    detections = network.forward() # forward() : 네트워크 실행(얼굴 인식)
    
    faces = [] # 얼굴 목록
    locations = [] # 좌표 목록
    predicts = [] # 확률 목록
    
    # 얼굴 인식을 위한 반복
    for i in range(0, detections.shape[2]):
        # 얼굴 인식 확률 추출
        confidence = detections[0, 0, i, 2]
        
        # 얼굴 인식 확률이 최소 확률보다 큰 경우
        if confidence > minimum_confidence:
            # bounding box 위치 계산
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            
            # bounding box 가 전체 좌표 내에 있는지 확인
            (startX, startY) = (max(0, startX), max(0, startY))
            (endX, endY) = (min(w - 1, endX), min(h - 1, endY))
            
            # 인식된 얼굴 추출 후 전처리
            face = frame[startY:endY, startX:endX] # 인식된 얼굴 추출
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB) # RGB 변환
            face = cv2.resize(face, (224, 224)) # 이미지(얼굴) 크기 조정
            face = img_to_array(face) # 이미지(얼굴)를 numpy 배열로 변환
            face = preprocess_input(face) # 모델에 필요한 형식에 이미지(얼굴)를 적절하게 맞추기위한 함수(전처리)
            
            # 전처리된 얼굴 이미지 및 좌표 목록 추가(업데이트)
            faces.append(face)
            locations.append((startX, startY, endX, endY))
            
    # 얼굴이 1개 이상 감지된 경우
    if len(faces) > 0:
        # 얼굴 목록을 numpy 배열로 변환
        faces = np.array(faces, dtype="float32")
        # predict() : 입력 dataset 에 대한 모델의 예측값
        predicts = model.predict(faces, batch_size=32)
    
    # 얼굴 위치 및 확률 목록 반환
    return (locations, predicts)

# 실행을 할 때 인자값 추가
ap = argparse.ArgumentParser() # 인자값을 받을 인스턴스 생성
# 입력받을 인자값 등록
ap.add_argument("-i", "--input", type=str, help="input 비디오 경로")
ap.add_argument("-o", "--output", type=str, help="output 비디오 경로") # 비디오 저장 경로
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

# input 비디오 경로가 제공되지 않은 경우 webcam
if not args.get("input", False):
    print("[webcam 시작]")
    vs = cv2.VideoCapture(0)

# input 비디오 경로가 제공된 경우 video
else:
    print("[video 시작]")
    vs = cv2.VideoCapture(args["input"])

# 인식할 최소 확률
minimum_confidence = 0.5

writer = None

# fps 정보 초기화
fps = FPS().start()

# 비디오 스트림 프레임 반복
while True:
    # 프레임 읽기
    ret, frame = vs.read()
    
    # 읽은 프레임이 없는 경우 종료
    if args["input"] is not None and frame is None:
        break
    
    # 프레임 resize
    frame = imutils.resize(frame, width=500)
    
    # 얼굴을 인식 및 Mask Detector 수행
    (locations, predicts) = mask_detector(frame, network, model)

    # 인식된 얼굴 수 만큼 반복
    for (box, predict) in zip(locations, predicts):
        (startX, startY, endX, endY) = box # bounding box 위치
        # mask : 마스크 착용 확률
        # without_mask : 마스크 미착용 확률
        (mask, without_mask) = predict # 확률
        
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
        cv2.putText(frame, label, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
        cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
    
    # 프레임 출력
    cv2.imshow("Mask Detector", frame)
    key = cv2.waitKey(1) & 0xFF
    
    # 'q' 키를 입력하면 종료
    if key == ord("q"):
        break
    
    # fps 정보 업데이트
    fps.update()
    
    # output video 설정
    if args["output"] != "" and writer is None:
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        writer = cv2.VideoWriter(args["output"], fourcc, 25, (frame.shape[1], frame.shape[0]), True)
    
    # 비디오 저장
    if writer is not None:
        writer.write(frame)

# fps 정지 및 정보 출력
fps.stop()
print("[재생 시간 : {:.2f}초]".format(fps.elapsed()))
print("[FPS : {:.2f}]".format(fps.fps()))

# 종료
vs.release()
cv2.destroyAllWindows()
