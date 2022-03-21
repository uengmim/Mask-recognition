# 필요한 패키지 import
import os # 운영체제 기능 모듈
import numpy as np # 파이썬 행렬 수식 및 수치 계산 처리 모듈
from PIL import Image # PIL(Python Image Library) : 파이썬 이미지 라이브러리

# 마스크 착용 이미지 생성
def create_mask(image, mask_image):
    image_path = image # 이미지 경로
    mask_path = mask_image # 마스크 이미지 경로
    model = "hog" # hog(기본값) : 비교적 덜 정확하지만 cpu에서도 빠름(gpu를 사용 가능한 경우 cnn)
    FaceMasker(image_path, mask_path, model).main() # FaceMasker의 main() 실행

class FaceMasker :
    # 주요 얼굴 특징(콧등, 턱)
    KEY_FACIAL_FEATURES = ('nose_bridge', 'chin')
    
    # 초기화
    def __init__(self, face_path, mask_path, model='hog') :
        self.face_path = face_path
        self.mask_path = mask_path
        self.model = model
    
    def main(self) :
        import face_recognition # 얼굴 특성 정보 추출(얼굴 인식) 모듈
        
        # face_recognition.load_image_file(이미지 파일 경로) : 이미지 파일을 numpy 배열로 로드
        load_image_file = face_recognition.load_image_file(self.face_path)
        # face_recognition.face_locations(이미지(numpy 배열), 모델) : 이미지에서 사람 얼굴의 bounding boxes 반환
        face_locations = face_recognition.face_locations(load_image_file, model=self.model)
        # face_recognition.face_landmarks(검색할 이미지(numpy 배열), 사람 얼굴 위치(bounding boxes) 목록) : 이미지의 각 얼굴의 특징 위치(눈, 코, 입 등) 목록
        face_landmarks = face_recognition.face_landmarks(load_image_file, face_locations)
        
        # Image.fromarray(이미지(numpy 배열)) : 버퍼 프로토콜을 사용하여 numpy 배열을 이미지로 변환
        self._face_image = Image.fromarray(load_image_file)
        # Image.open(이미지 파일 경로) : 이미지 파일 로드
        self._mask_image = Image.open(self.mask_path)

        found_face = False # 얼굴이 있는지 없는지 판별

        # 얼굴 인식 목록 수 만큼 반복
        for face_landmark in face_landmarks :
            skip = False # 얼굴 특징이 요구 사항을 충족하는지 확인
            
            # 주요 얼굴 특징 수 만큼 반복(콧등, 턱 : 2)
            for facial_feature in self.KEY_FACIAL_FEATURES :
                # 주요 얼굴 특징(콧등, 턱)이 아닌 경우
                if facial_feature not in face_landmark :
                    skip = True
                    break
            
            # 요구 사항을 충족하지 않는 경우
            if skip :
                continue
            
            found_face = True # 얼굴이 있음

            # 마스크를 얼굴에 착용
            self.mask_face(face_landmark)

        # 얼굴이 있는 경우
        if found_face:
            # 이미지 저장
            self._save()
        # 얼굴이 없는 경우
        else:
            print('얼굴을 찾지 못함')
            print(self.face_path)

    # 마스크를 얼굴에 착용
    def mask_face(self, face_landmark : dict):
        # 콧등 영역 추출
        nose_bridge = face_landmark['nose_bridge']
        nose_point = nose_bridge[len(nose_bridge) * 1 // 4]
        nose = np.array(nose_point)

        # 턱 영역 추출
        chin = face_landmark['chin']
        chin_len = len(chin)
        chin_bottom_point = chin[chin_len // 2]
        chin_bottom = np.array(chin_bottom_point)
        chin_left_point = chin[chin_len // 8]
        chin_right_point = chin[chin_len * 7 // 8]

        # 얼굴 크기에 맞게 마스크 분할 및 크기 조절
        width = self._mask_image.width
        height = self._mask_image.height
        width_ratio = 1.2 # 마스크 너비
        # np.linalg.norm : 벡터 공간의 유클리디안 거리 계산
        mask_height = int(np.linalg.norm(nose - chin_bottom)) # 마스크의 높이

        # 마스크 왼쪽
        mask_left_image = self._mask_image.crop((0, 0, width // 2, height)) # crop(좌표) : 이미지를 부분적으로 추출
        mask_left_width = self.get_distance(chin_left_point, nose_point, chin_bottom_point) # 왼쪽 턱 너비
        mask_left_width = int(mask_left_width * width_ratio) # 왼쪽 너비 * 1.2 = 마스크 왼쪽 너비
        mask_left_image = mask_left_image.resize((mask_left_width, mask_height)) # 마스크 왼쪽 크기 조절

        # 마스크 오른쪽
        mask_right_image = self._mask_image.crop((width // 2, 0, width, height)) # crop(좌표) : 이미지를 부분적으로 추출
        mask_right_width = self.get_distance(chin_right_point, nose_point, chin_bottom_point) # 오른쪽 턱 너비
        mask_right_width = int(mask_right_width * width_ratio) # 오른쪽 너비 * 1.2 = 마스크 오른쪽 너비
        mask_right_image = mask_right_image.resize((mask_right_width, mask_height)) # 마스크 오른쪽 크기 조절

        # 변경된 크기의 왼쪽 + 오른쪽 마스크 합치기
        size = (mask_left_image.width + mask_right_image.width, mask_height) # 변경된 마스크 크기
        mask_image = Image.new('RGBA', size) # Image.new(모드, 크기) : 새로운 이미지 생성
        # paste(이미지, 좌표, 이미지) : 이미지 합치기
        mask_image.paste(mask_left_image, (0, 0), mask_left_image)
        mask_image.paste(mask_right_image, (mask_left_image.width, 0), mask_right_image)

        # 마스크 회전
        angle = np.arctan2(chin_bottom_point[1] - nose_point[1], chin_bottom_point[0] - nose_point[0]) # np.arctan2() : 사분면을 올바르게 선택하는 요소별 아크 탄젠트
        rotated_mask_image = mask_image.rotate(angle, expand=True) # rotate(각도, expand : 출력 크기에 맞게 조정) : 이미지 회전

        # 마스크 위치 계산
        center_x = (nose_point[0] + chin_bottom_point[0]) // 2
        center_y = (nose_point[1] + chin_bottom_point[1]) // 2

        offset = mask_image.width // 2 - mask_left_image.width
        radian = angle * np.pi / 180
        box_x = center_x + int(offset * np.cos(radian)) - rotated_mask_image.width // 2 # cos(각도) : cosine
        box_y = center_y + int(offset * np.sin(radian)) - rotated_mask_image.height // 2 # sin(각도) : sine

        # 마스크 추가
        self._face_image.paste(mask_image, (box_x, box_y), mask_image)

    # 이미지 저장
    def _save(self):
        image_path_splits = os.path.splitext(self.face_path) # os.path.splitext(파일 경로) : 파일의 확장자 추출
        number = image_path_splits[0].split('_') # 이미지 이름 숫자 추출
        with_mask_image_path = '../dataset/with_mask/with_mask_' + number[3] + image_path_splits[1] # 이미지 저장 경로
        self._face_image.save(with_mask_image_path) # 이미지 저장
        print('저장 경로 :', with_mask_image_path)

    @staticmethod
    # 거리 계산 함수
    def get_distance(point, line_point1, line_point2):
        # np.abs : 절대값으로 변경
        # np.sqrt : 제곱근
        distance = np.abs((line_point2[1] - line_point1[1]) * point[0] + (line_point1[0] - line_point2[0]) * point[1] + (line_point2[0] - line_point1[0]) * line_point1[1] + (line_point1[1] - line_point2[1]) * line_point1[0]) / \
                   np.sqrt((line_point2[1] - line_point1[1]) * (line_point2[1] - line_point1[1]) + (line_point1[0] - line_point2[0]) * (line_point1[0] - line_point2[0]))
        return int(distance)

# 메인 함수(인터프리터에서 직접 실행했을 경우에만 if 문 내의 코드 실행)
if __name__ == '__main__':
    create_mask(image, mask_image)
