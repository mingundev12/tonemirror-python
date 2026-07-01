import cv2
import numpy as np

from validator.face_mesh_validator import FaceMeshValidator

class PhotoQualityValidator:

    def __init__(self):
        self.face_mesh_validator = FaceMeshValidator()

    # 흐림 처리 ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ

    def validate_blur(self, image):
        gray = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2GRAY
        )

        score = cv2.Laplacian(
            gray,
            cv2.CV_64F
        ).var()

        if score < 80:
            return False, f"사진이 흐립니다. ({score:.1f})"
        
        return True, f"정상 ({score:.1f})"
    
    # 노출 처리 ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ

    def validate_exposure(self, skin_region):

        gray = cv2.cvtColor(
            skin_region,
            cv2.COLOR_BGR2GRAY
        )

        pixels = gray[gray > 0]

        if len(pixels) == 0:
            return False, "피부 영역 없음"

        mean_brightness = np.mean(pixels)

        if mean_brightness > 210:
            return False, (
                f"과다 노출 ({mean_brightness:.1f})"
            )

        if mean_brightness < 70:
            return False, (
                f"노출 부족 ({mean_brightness:.1f})"
            )

        return True, (
            f"정상 노출 ({mean_brightness:.1f})"
        )
    
    # 평균 밝기 계산 ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ

    def get_mean_brightness(self, region):

        gray = cv2.cvtColor(
            region,
            cv2.COLOR_BGR2GRAY
        )

        pixels = gray[gray > 0]

        if len(pixels) == 0:
            return 0
        
        return np.mean(pixels)
    
    # 그림자 검사 ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ

    def validate_shadow(self, left_cheek_region, right_cheek_region):

        left_brightness = self.get_mean_brightness(
            left_cheek_region
        )

        right_brightness = self.get_mean_brightness(
            right_cheek_region
        )

        diff = abs(
            left_brightness - right_brightness
        )

        if diff > 25:
            return(False, f"그림자가 심합니다. 밝기 차이 : {diff:.1f}")
        
        return(True, f"정상. 밝기 차이 : {diff:.1f}")
    
    # 정면 얼굴 검사 ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ
    # 정면이 아닐 경우 왼쪽으로 돌리면 left거리 > right거리 , 오른쪽으로 돌리면 right거리 > left거리

    def validate_pose(self, image, landmarks):
        
        height, width = image.shape[:2]

        nose_x = int(landmarks.landmark[1].x * width)

        left_x = int(landmarks.landmark[234].x * width)

        right_x = int(landmarks.landmark[454].x * width)

        left_dist = nose_x - left_x
        right_dist = right_x - nose_x

        if right_dist == 0:
            return False, "얼굴 방향 계산 실패"
        
        ratio = left_dist / right_dist

        if ratio < 0.75 or ratio > 1.25:
            return False, (f"정면 얼굴이 아닙니다. {ratio:.2f}")
        
        return True, (f"정면 얼굴 확인 {ratio:.2f}")
    
    # 얼굴 크기 검사 ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ

    def validate_face_size(self, image, landmarks):
        
        height, width = image.shape[:2]

        left_x = int(landmarks.landmark[234].x * width)
        right_x = int(landmarks.landmark[454].x * width)
        top_y = int(landmarks.landmark[10].y * height)
        bottom_y = int(landmarks.landmark[152].y * height)

        face_width = right_x - left_x
        face_height = bottom_y - top_y

        face_area = (face_width * face_height)

        image_area = width * height

        ratio = (face_area / image_area)

        if ratio < 0.12:
            return(False, f"얼굴이 너무 작습니다. ({ratio:.2f})")
        
        return(True, f"얼굴 크기 정상. ({ratio:.2f})")
    
    # 얼굴 기울기 검사 ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ

    def validate_tilt(self, image, landmarks):
        
        height, width = image.shape[:2]

        left_eye = landmarks.landmark[33]
        right_eye = landmarks.landmark[263]

        left_x = int(left_eye.x * width)
        left_y = int(left_eye.y * height)

        right_x = int(right_eye.x * width)
        right_y = int(right_eye.y * height)

        dy = right_y - left_y
        dx = right_x - left_x

        angle = np.degrees(
            np.arctan2(dy, dx)
        )

        if abs(angle) > 8:
            return(False, f"얼굴이 기울어져 있습니다. ({angle:.1f})")
        
        return(True, f"얼굴 기울기 정상 ({angle:.1f})")
    
    # 얼굴 잘림 검사 ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ

    def validate_face_crop(self, image, landmarks):
        
        height, width = image.shape[:2]

        left_x = int(landmarks.landmark[234].x * width)
        right_x = int(landmarks.landmark[454].x * width)
        top_y = int(landmarks.landmark[10].y * height)
        bottom_y = int(landmarks.landmark[152].y * height)

        margin = 20

        if top_y < margin:
            return False, "이마가 잘렸습니다."
        
        if bottom_y > height - margin:
            return False, "턱이 잘렸습니다."
        
        if left_x < margin:
            return False, "왼쪽 얼굴이 잘렸습니다."
        
        if right_x > width - margin:
            return False, "오른쪽 얼굴이 잘렸습니다."
        
        return True, "얼굴 잘림 없음"
    
    # 눈 감음 검사 ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ

    def validate_eye_open(self, image, landmarks):
        
        height, width = image.shape[:2]

        # 왼쪽 눈
        left_outer = landmarks.landmark[33]
        left_inner = landmarks.landmark[133]

        left_top = landmarks.landmark[159]
        left_bottom = landmarks.landmark[145]

        # 오른쪽 눈
        right_outer = landmarks.landmark[362]
        right_inner = landmarks.landmark[263]

        right_top = landmarks.landmark[386]
        right_bottom = landmarks.landmark[374]

        # 픽셀 좌표 변환
        def point(lm):
            return np.array([
                lm.x * width,
                lm.y * height
            ])
        
        # 왼쪽 EAR
        left_width = np.linalg.norm(
            point(left_outer) - point(left_inner)
        )

        left_height = np.linalg.norm(
            point(left_top) - point(left_bottom)
        )

        left_ear = left_height / left_width

        # 오른쪽 EAR
        right_width = np.linalg.norm(
            point(right_outer) - point(right_inner)
        )

        right_height = np.linalg.norm(
            point(right_top) - point(right_bottom)
        )

        right_ear = right_height / right_width

        ear = (left_ear + right_ear) / 2

        if ear < 0.20:
            return False, f"두 눈이 감겨있습니다. ({ear:.2f})"
        
        return True, f"눈 뜸 확인. ({ear:.2f})"
    
    # 앞머리 검사 ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ

    def validate_forehead_visible(self, image, landmarks):
        
        height, width = image.shape[:2]

        forehead = landmarks.landmark[10]

        x = int(forehead.x * width)
        y = int(forehead.y * height)

        roi = image[
            y:y+80,
            max(0, x-80):x+80
        ]

        if roi.size == 0:
            return False, "이마 영역 추출 실패"
        
        gray = cv2.cvtColor(
            roi,
            cv2.COLOR_BGR2GRAY
        )

        dark_ratio = (
            np.sum(gray < 60) / gray.size
        )

        if dark_ratio > 0.30:
            return False, f"앞머리가 이마를 가랍니다. ({dark_ratio:.2f})"
        
        return True, f"이마 노출 정상 ({dark_ratio:.2f})"
    
    # validate 호출 ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ

    # blur 검사
    def validate_blur_only(self, image):

        passed, message = self.validate_blur(image)

        if not passed:
            return {
                "passed": False,
                "message": message
            }
        
        return {
            "passed": True,
            "message": "Blur 검사 통과"
        }
        
    # 얼굴 품질 검사
    def validate_face(self, image):

        # FaceMesh 1회 실행
        landmarks = self.face_mesh_validator.get_landmarks(image)

        if landmarks is None:
            return {
                "passed": False,
                "message": "얼굴 랜드마크 검출 실패"
            }

        checks = [
            self.validate_pose(image, landmarks),
            self.validate_face_size(image, landmarks),
            self.validate_tilt(image, landmarks),
            self.validate_face_crop(image, landmarks),
            self.validate_eye_open(image, landmarks),
            self.validate_forehead_visible(image, landmarks)
        ]

        for passed, message in checks:

            if not passed:
                return {
                    "passed": False,
                    "message": message
                }
            
        return {
            "passed": True,
            "message": "사진 품질 검사 통과"
        }

