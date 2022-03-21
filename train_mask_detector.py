# 필요한 패키지 import
# tensorflow : 구글에서 만든 머신러닝 및 딥러닝 프로그램을 쉽게 구현할 수 있도록 다양한 기능을 제공해 주는 라이브러리
# keras : 오픈 소스 신경망 라이브러리(텐서플로우 2.0.0 이상의 버전을 설치했을 경우 텐서플로우에 keras가 내장)
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input # MobileNetV2 모델에 필요한 형식에 이미지를 적절하게 맞추기위한 함수(전처리)
from tensorflow.keras.preprocessing.image import img_to_array # 이미지를 numpy 배열로 변환
from tensorflow.keras.preprocessing.image import load_img # 이미지 로드
from tensorflow.keras.utils import to_categorical # One-Hot Encoding 처리
from tensorflow.keras.preprocessing.image import ImageDataGenerator # 데이터를 변형시켜 새로운 학습 dataset을 생성
# MobileNetV2
# - 모바일이나, 임베디드에서도 실시간 작동할 수 있게 모델이 경량화(메모리와 연산량 감소)
# - 정확도 또한 많이 떨어지지 않아 속도와 정확도 사이의 트레이드-오프(trade-off) 문제를 어느 정도 해결
# - 트레이드-오프(trade-off) 문제 : 한 부분의 성능을 높이면 다른 부분의 성능이 낮아지는 문제
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Input # 입력 데이터의 모양을 모델에 알려주는 역할(Keras tensor 를 인스턴스화)
from tensorflow.keras.layers import Conv2D # 이미지 특징 추출
from tensorflow.keras.layers import AveragePooling2D # 추출한 2D feature map의 차원을 다운 샘플링하여 연산량을 감소시키고 주요한 특징 벡터를 추출 - Average Pooling : 각 filter에서 다루는 이미지 패치에서 모든 값의 평균을 반환
from tensorflow.keras.layers import Flatten # 2차원 데이터로 이루어진 추출된 특징을 Dense Layer 에서 학습하기 위해 1차원 데이터로 변경
from tensorflow.keras.layers import Dense # 입력과 출력을 모두 연결(Fully Connected Layer)
from tensorflow.keras.layers import Dropout # 망의 크기가 커질 경우 Overfitting 문제를 피하기 위해 학습을 할 때 일부 뉴런을 생략하여 학습
from tensorflow.keras.models import Model # 학습 및 추론 기능을 가진 객체로 Layer를 모델링
# Nadam(Nesterov-accelerated Adaptive Moment Estimation) : NAG(Nesterov Accelarated Gradient) + Adam(Adaptive Moment Estimation)
# NAG(Nesterov accelarated gradient)
# - momentum 이 이동시킬 방향으로 미리 이동해서 gradient 를 계산(불필요한 이동을 줄이는 효과) - 정확도 개선
# - momentum : 경사 하강법에 관성을 더해주는 것으로, 매번 계산된 기울기에 과거 이동했던 방향을 기억하면서 그 방향으로 일정 값을 추가적으로 더해주는 방식
# Adam(Adaptive Moment Estimation) : momentum + RMSProp (정확도와 보폭 크기 개선)
# - RMSProp : Adagrad 의 보폭 민감도를 보완한 방법(보폭 크기 개선)
# - Adagrad : 변수의 업데이트가 잦으면 학습률을 작게하여 이동 보폭을 조절하는 방법(보폭 크기 개선)
# optimizer : 모델을 컴파일하기 위해 필요한 최적화 알고리즘
from tensorflow.keras.optimizers import Nadam

# sklearn : 머신러닝 분석을 할 때 유용하게 사용할 수 있는 라이브러리(머신러닝 모듈로 구성)
from sklearn.preprocessing import LabelBinarizer # 레이블 이진화
from sklearn.model_selection import train_test_split # 학습 dataset 과 테스트 dataset 분리
from sklearn.metrics import classification_report # 분류 지표 텍스트

from imutils import paths # 파이썬 OpenCV가 제공하는 기능 중 복잡하고 사용성이 떨어지는 부분을 보완(이미지 또는 비디오 스트림 파일 처리 등)
import matplotlib.pyplot as plt # 데이터를 차트나 그래프로 시각화할 수 있는 라이브러리
import numpy as np # 파이썬 행렬 수식 및 수치 계산 처리 모듈
import os # 운영체제 기능 모듈

dataset = './dataset' # 학습 dataset 경로
plot = 'result_plot.jpg' # matplotlib 를 사용한 학습 오차 및 정확도 그래프
model_name = 'mask_detector.h5' # Face Mask Detector 학습 결과 

init_learning_rate = 1e-4 # 초기 학습률(1 * (10**(-4))) : e는 Exponential(지수)의 약자
epochs = 20 # epoch 수 : 전체 dataset에 대해 학습할 횟수
batch_size = 32 # batch size : sample 데이터 중 한 번에 네트워크에 넘겨주는 데이터의 수

print("[이미지(dataset) 로딩]")
image_paths = list(paths.list_images(dataset)) # dataset 경로에 모든 이미지 가져오기
data = [] # 데이터 목록
labels = [] # 레이블 목록

# dataset 경로의 이미지 수 만큼 반복
for image_path in image_paths:
    # 파일 이름에서 class 레이블 추출(without_mask, with_mask)
    label = image_path.split(os.path.sep)[-2]

    image = load_img(image_path, target_size=(224, 224)) # 입력 이미지를 224 x 224 크기로 로드
    image = img_to_array(image) # 이미지를 numpy 배열로 변환
    image = preprocess_input(image) # 모델에 필요한 형식에 이미지를 적절하게 맞추기위한 함수(전처리)

    # 전처리된 이미지(데이터 목록) 및 레이블 목록 추가(업데이트)
    data.append(image)
    labels.append(label)

# 데이터와 레이블을 numpy 배열로 변환
data = np.array(data, dtype="float32")
labels = np.array(labels)

# class 레이블 One-Hot Encoding : 단어 집합의 크기를 벡터의 차원으로 하고 표현하고 싶은 단어의 인덱스에 1, 다른 인덱스에는 0 을 부여하는 단어의 벡터 표현 방식
label_binarizer = LabelBinarizer() # LabelBinarizer 객체 생성
labels = label_binarizer.fit_transform(labels) # 레이블 이진화(fit : 평균과 표준편차 계산, transform : 정규화)
labels = to_categorical(labels) # 이진화된 레이블 One-Hot Encoding 처리

# 학습 dataset 80%, 테스트 dataset 20% 로 분리
# stratify : dataset class 비율 유지
# random_state : 데이터 분할시 shuffle(데이터 섞기)이 이루어지는데 이를 위한 Seed 값(1 또는 다른 수를 적용하면 코드를 실행할 때마다 결과 동일, 적용하지 않은 경우 코드를 실행할 때마다 다른 결과)
(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.2, stratify=labels, random_state=42)

# 데이터 증가를 위한 학습 이미지 생성기 구성
# rotation_range : 지정된 각도 범위내에서 임의로 원본 이미지를 회전
# width_shift_range : 지정된 수평 방향 이동 범위내에서 임의로 원본 이미지를 이동
# height_shift_range : 지정된 수직 방향 이동 범위내에서 임의로 원본 이미지를 이동
# shear_range : 지정된 밀림 강도 범위내에서 임의로 원본 이미지를 변경
# zoom_range : 지정된 확대/축소 범위내에서 임의로 원본 이미지를 확대/축소
# horizontal_flip : 수평 방향으로 뒤집기
# vertical_flip : 수직 방향으로 뒤집기
# fill_mode : 이미지를 회전, 이동하거나 축소할 때 생기는 공간을 채우는 방식
# - constant : 임의의 값              [5555 | 1234(원본) | 5555]
# - nearest : 가장 가까운 값(default) [1111 | 1234(원본) | 4444]
# - reflect : 반사된 값               [4321 | 1234(원본) | 4321]
# - wrap : 모든 값(순서대로)          [1234 | 1234(원본) | 1234]
image_data_generator = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode="nearest")

# MobileNetV2 로드
# weights : 로드할 가중치 파일(imagenet : 사전 학습된 ImageNet 가중치)
# include_top : 네트워크 상단의 Fully-Connected Layer 포함 여부(default : True)
# input_tensor : 모델에 대한 이미지 입력으로 사용할 Keras tensor
# input_shape : 입력 이미지 해상도가 (224, 224, 3)이 아닌 모델을 사용하려는 경우 지정(include_top 이 False 인 경우)
# alpha : 네트워크의 너비를 제어
# - 1 보다 크면 필터 수가 비례적으로 감소
# - 1 보다 작으면 필터 수가 비례적으로 증가
# - 1 인 경우 기본 필터 수
# pooling : 특성 추출을 위한 pooling 모드(include_top 이 False 인 경우)
# - None : 모델의 출력이 마지막 Convolutional Layer 의 4D tensor 가 출력
# - avg : average pooling 적용
# - max : max pooling 적용
# classes : 이미지를 분류하기 위한 선택적 class 수(include_top 이 True 이며 weights 가 특정되지 않은 경우)
# classifier_activation : 최상위 Layer 에서 사용할 활성화 함수(include_top 이 True 인 경우)
input_model = MobileNetV2(
        weights="imagenet", 
        include_top=False,
        input_tensor=Input(shape=(224, 224, 3)))

# 새로운 Fully Connected Layer 구성
output_model = input_model.output # 로드한 MobileNetV2 의 출력을 사용
# Conv2D
# 첫 번째 인자 : Convolution filter 수
# 두 번째 인자 : Convolution filter (행, 열)
# padding : 경계 처리 방법
# - valid : 유효한 영역만 출력(출력 이미지 크기 < 입력 이미지 크기)
# - same : 출력 이미지 크기가 입력 이미지 크기와 동일
# input_shape : sample 수를 제외한 입력 형태를 정의(모델에서 첫 Layer 일 때만 정의)
# activation : 활성화 함수
# - linear : 입력 뉴련과 가중치로 계산된 결과값이 그대로 출력(default)
# - relu : 0 보다 작은 값일 경우 0을 반환, 0 보다 큰 값일 경우 그 값을 그대로 반환
# - sigmoid : 입력 값을 0 보다 크고 1 보다 작은 미분 가능한 수로 반환
# - softmax : 입력받은 값으로 분류할 class의 출력을 0 ~ 1 사이의 값으로 정규화(출력 값들의 총 합이 1이 되는 특성)
output_model = Conv2D(32, (5, 5), padding="same", activation="relu")(output_model) # Convolution Layer
# AveragePooling2D
# pool_size : pooling 을 적용할 각 filter 크기
# strides : stride 간격
# padding : 경계 처리 방법
# - valid : 유효한 영역만 출력(출력 이미지 크기 < 입력 이미지 크기)
# - same : 출력 이미지 크기가 입력 이미지 크기와 동일
output_model = AveragePooling2D(pool_size=(5, 5), strides=1, padding="same")(output_model) # Average Pooling Layer
output_model = Flatten(name="flatten")(output_model) # 2차원 데이터로 이루어진 추출된 특징을 Dense Layer 에서 학습하기 위해 1차원 데이터로 변경
# Dense
# 첫 번째 인자 : 출력 뉴런의 수
# input_dim : 입력 뉴런의 수
# init : 가중치 초기화 방법(uniform : 균일 분포, normal : 가우시안 분포)
# activation : 활성화 함수
# - linear : 입력 뉴련과 가중치로 계산된 결과값이 그대로 출력(default)
# - relu : 0 보다 작은 값일 경우 0을 반환, 0 보다 큰 값일 경우 그 값을 그대로 반환
# - sigmoid : 입력 값을 0 보다 크고 1 보다 작은 미분 가능한 수로 반환
# - softmax : 입력받은 값으로 분류할 class의 출력을 0 ~ 1 사이의 값으로 정규화(출력 값들의 총 합이 1이 되는 특성)
output_model = Dense(32, activation="relu")(output_model) # 출력 뉴런이 32 개 이고 활성화 함수가 relu
output_model = Dense(64, activation="relu")(output_model) # 출력 뉴런이 64 개 이고 활성화 함수가 relu
output_model = Dropout(0.5)(output_model) # 50% Dropout 적용(dropout을 적용할 비율)
output_model = Dense(32, activation="relu")(output_model) # 출력 뉴런이 32 개 이고 활성화 함수가 relu
output_model = Dense(2, activation="softmax")(output_model) # [Output Layer] 출력 뉴런이 2 개 이고 활성화 함수가 softmax

# 학습할 모델 생성
# inputs : 모델의 입력
# outputs : 모델의 출력
# name : 모델의 이름
model = Model(inputs=input_model.input, outputs=output_model)

# 기존 MobileNetV2 모델의 모든 Layer 를 반복하고 고정(첫 번째 학습 과정 동안 업데이트되지 안도록 함)
for layer in input_model.layers:
    layer.trainable = False

# Nadam optimizer 설정
# lr : 학습률(default : 0.001)
# beta_1 : 첫 번째 모멘트 추정치에 대한 감소율(default : 0.9)
# beta_2 : 두 번째 모멘트 추정치에 대한 감소율(default : 0.999)
# epsilon : 학습 속도(default : 1e-7)
# decay : 업데이트마다 적용되는 학습률의 감소율
# optimizer : 모델을 컴파일하기 위해 필요한 최적화 알고리즘
optimizer = Nadam(lr=init_learning_rate, decay=init_learning_rate / epochs)

print("[모델 컴파일]")
# Model.compile() : 모델 컴파일
# optimizer : 적용할 optimizer
# loss : 오차(loss : 예측값과 실제값 간의 차이)를 표현
# - binary_crossentropy : 카테고리가 2개인 경우
# - categorical_crossentropy : 카테고리가 3개 이상인 경우
# metrics : 학습이 잘 이루어지는지 판단(평가)하는 기준 목록
# - accuracy : 정확도
# - mse : 평균 제곱근 오차(Mean Squared Error)
model.compile(loss="binary_crossentropy", optimizer=optimizer, metrics=["accuracy"])

print("[모델 학습]")
# Model.fit() : 모델 학습
# 첫 번째 인자 : 입력 데이터
# - ImageDataGenerator.flow() : 랜덤하게 변형된 학습 dataset 생성
# steps_per_epoch : ImageDataGenerator 로부터 얼마나 많은 sample 을 추출할 것인지
# validation_data : 한 번의 epoch 를 실행할 때마다 학습 결과를 확인할 Validation 데이터
# validation_steps : 학습 종료 전 검증할 단계(sample)의 총 개수
# epochs : 전체 dataset에 대해 학습할 횟수
train = model.fit(
        image_data_generator.flow(trainX, trainY, batch_size=batch_size),
        steps_per_epoch=len(trainX) // batch_size,
        validation_data=(testX, testY),
        validation_steps=len(testX) // batch_size,
        epochs=epochs)

print("[모델 평가]")
# Model.predict() : 테스트 입력 dataset 에 대한 모델의 출력값 확인
predict = model.predict(testX, batch_size=batch_size)
# numpy.argmas : 최대값 index 반환
# - axis : 계산할 기준(0 : 열, 1 : 행)
predict_index = np.argmax(predict, axis=1)
# 분류 결과 출력
print(classification_report(testY.argmax(axis=1), predict_index, target_names=label_binarizer.classes_))

print("[모델 저장]")
# Model.save() : 모델 아키텍처 및 가중치 저장
# - save_format : 저장 형식
model.save(model_name, save_format="h5")

# 학습 오차 및 정확도 그래프
n = epochs
plt.style.use("ggplot") # style.use() : 스타일 적용
plt.figure() # figure() : 새로운 figure 생성
# plot() : 그래프 그리기
# 첫 번째 인자 : X 축 데이터
# - np.arange() : 인자로 받는 값 만큼 1씩 증가하는 1차원 배열 생성
# 두 번째 인자 : Y 축 데이터
# - Model.fit.history : 학습 오차 및 정확도, 검증 오차 및 정확도
# label : 범례
plt.plot(np.arange(0, n), train.history["loss"], label="train_loss") # epoch 마다 학습 오차
plt.plot(np.arange(0, n), train.history["val_loss"], label="val_loss") # epoch 마다 검증 오차
plt.plot(np.arange(0, n), train.history["acc"], label="train_accuracy") # epoch 마다 학습 정확도
plt.plot(np.arange(0, n), train.history["val_acc"], label="val_accuracy") # epoch 마다 검증 정확도
plt.title("Training Loss and Accuracy") # title() : 제목
plt.xlabel("epoch") # xlabel() : X 축 레이블
plt.ylabel("Loss / Accuracy") # ylabel() : Y 축 레이블
plt.legend(loc="lower left") # legend() : 범례 위치
plt.savefig(plot) # 그래프 이미지 파일로 저장
