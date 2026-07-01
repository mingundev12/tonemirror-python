import cv2
import mediapipe as mp


class FaceMeshValidator:

    def __init__(self):

        # FaceMesh 객체 생성
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=True,       # 이미지 한 장 처리
            max_num_faces=5,              # 얼굴 개수 확인을 위해 여러 명 탐지
            refine_landmarks=True,        # 눈, 입술 정밀도 향상
            min_detection_confidence=0.5
        )

    def get_landmarks(self, image):

        # BGR → RGB 변환
        rgb_image = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB
        )

        # FaceMesh 실행
        results = self.face_mesh.process(rgb_image)

        # 얼굴이 검출되지 않은 경우
        if not results.multi_face_landmarks:
            return None

        # 얼굴이 1명이 아닌 경우
        if len(results.multi_face_landmarks) != 1:
            return None

        # 단일 얼굴 랜드마크 반환
        return results.multi_face_landmarks[0]