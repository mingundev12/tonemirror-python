import cv2
import numpy as np

from validator.face_mesh_validator import FaceMeshValidator

# 얼굴 외곽
FACE_OVAL = [
    10,338,297,332,284,251,389,356,
    454,323,361,288,397,365,379,378,
    400,377,152,148,176,149,150,136,
    172,58,132,93,234,127,162,21,
    54,103,67,109
]

# 눈
LEFT_EYE = [
    33,7,163,144,145,153,154,155,
    133,173,157,158,159,160,161,246
]

RIGHT_EYE = [
    362,382,381,380,374,373,390,249,
    263,466,388,387,386,385,384,398
]

# 입술
LIPS = [
    61,146,91,181,84,17,314,405,
    321,375,291,308,324,318,402,317,
    14,87,178,88,95,
    185,40,39,37,0,267,269,270,
    409,415,310,311,312,13,82,81,80,191
]

# 눈썹
LEFT_EYEBROW = [
    70,63,105,66,107,
    55,65,52,53,46
]

RIGHT_EYEBROW = [
    336,296,334,293,300,
    285,295,282,283,276
]

# 홍채
LEFT_IRIS = [
    474, 475, 476, 477
]

RIGHT_IRIS = [
    469, 470, 471, 472
]

class RegionExtractor:

    def __init__(self):
        
        self.face_mesh_validator = (FaceMeshValidator())

    # 랜드마크 -> polygon 함수 ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ

    def create_mask(
        self,
        points,
        face_landmarks,
        width,
        height
    ):
        
        polygon = []

        for idx in points:

            landmark = (face_landmarks.landmark[idx])

            x = int(landmark.x * width)
            y = int(landmark.y * height)

            polygon.append([x,y])

        polygon = np.array(polygon, np.int32)

        mask = np.zeros((height, width), dtype=np.uint8)

        cv2.fillPoly(
            mask,
            [polygon],
            255
        )

        return mask
    
    # 홍채 마스크 ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ

    def get_iris_mask(self, face_landmarks, width, height):

        iris_mask = np.zeros((height, width), dtype=np.uint8)

        for iris_points in [LEFT_IRIS, RIGHT_IRIS]:

            points = []

            for idx in iris_points:
                landmark = face_landmarks.landmark[idx]

                x = int(landmark.x * width)
                y = int(landmark.y * height)

                points.append((x,y))

            center_x = int(np.mean([p[0] for p in points]))
            center_y = int(np.mean([p[1] for p in points]))

            radius = max(
                5,
                int(
                    np.linalg.norm(
                        np.array(points[0]) - np.array(points[2])
                    ) / 2
                )
            )

            cv2.circle(
                iris_mask,
                (center_x, center_y),
                radius,
                255,
                -1
            )

        return iris_mask
    
    # 눈썹 마스크 ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ

    def get_eyebrow_hair_mask(self, image, eyebrow_mask):

        eyebrow_region = cv2.bitwise_and(
            image,
            image,
            mask = eyebrow_mask
        )

        eyebrow_gray = cv2.cvtColor(
            eyebrow_region,
            cv2.COLOR_BGR2GRAY
        )

        _, eyebrow_hair_mask = cv2.threshold(
            eyebrow_gray,
            0,
            255,
            cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )

        eyebrow_hair_mask = cv2.bitwise_and(
            eyebrow_hair_mask,
            eyebrow_mask
        )

        return eyebrow_hair_mask
    
    # 이마 마스크 ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ

    def get_forehead_sample_mask(self, face_landmarks, width, height):

        # 얼굴 중앙
        center_x = int(face_landmarks.landmark[10].x * width)
        top_y = int(face_landmarks.landmark[10].y * height)

        face_left = face_landmarks.landmark[234]
        face_right = face_landmarks.landmark[454]

        face_width = int(
            abs(face_right.x - face_left.x) * width
        )

        eyebrow_y = int(np.mean([
            face_landmarks.landmark[i].y * height
            for i in LEFT_EYEBROW + RIGHT_EYEBROW
        ]))

        forehead_height = eyebrow_y - top_y

        sample_top = top_y + int(forehead_height * 0.15)
        sample_bottom = top_y + int(forehead_height * 0.70)

        sample_width = int(face_width * 0.22)

        left_x = center_x - sample_width
        right_x = center_x + sample_width

        mask = np.zeros((height, width), dtype=np.uint8)

        cv2.rectangle(
            mask,
            (left_x, sample_top),
            (right_x, sample_bottom),
            255,
            -1
        )

        return mask
    
    # 왼쪽 볼 마스크 ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ

    def get_left_cheek_mask(self, face_landmarks, width, height):

        cheek = face_landmarks.landmark[205]

        cx = int(cheek.x * width)
        cy = int(cheek.y * height)

        face_left = face_landmarks.landmark[234]
        face_right = face_landmarks.landmark[454]

        face_width = int(
            abs(face_right.x - face_left.x) * width
        )

        mask = np.zeros((height, width), dtype=np.uint8)

        cv2.ellipse(
            mask,
            (cx, cy),
            (
                int(face_width * 0.13),
                int(face_width * 0.10)
            ),
            0,
            0,
            360,
            255,
            -1
        )

        return mask

    # 오른쪽 볼 마스크 ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ

    def get_right_cheek_mask(self, face_landmarks, width, height):

        cheek = face_landmarks.landmark[425]

        cx = int(cheek.x * width)
        cy = int(cheek.y * height)

        face_left = face_landmarks.landmark[234]
        face_right = face_landmarks.landmark[454]

        face_width = int(
            abs(face_right.x - face_left.x) * width
        )

        mask = np.zeros((height, width), dtype=np.uint8)

        cv2.ellipse(
            mask,
            (cx, cy),
            (
                int(face_width * 0.13),
                int(face_width * 0.10)
            ),
            0,
            0,
            360,
            255,
            -1
        )

        return mask
    
    # 피부 region 생성 ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ
    
    def extract(self, image):

        height, width = image.shape[:2]

        face_landmarks = (self.face_mesh_validator.get_landmarks(image))

        if face_landmarks is None:
            return None
        
        # 마스크 생성 ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ

        # 얼굴
        face_mask = self.create_mask(
            FACE_OVAL,
            face_landmarks,
            width,
            height
        )

        # 눈
        left_eye_mask = self.create_mask(
            LEFT_EYE,
            face_landmarks,
            width,
            height
        )
        right_eye_mask = self.create_mask(
            RIGHT_EYE,
            face_landmarks,
            width,
            height
        )

        # 입술
        lip_mask = self.create_mask(
            LIPS,
            face_landmarks,
            width,
            height
        )

        # 눈썹
        left_eyebrow_mask = self.create_mask(
            LEFT_EYEBROW,
            face_landmarks,
            width,
            height
        )
        right_eyebrow_mask = self.create_mask(
            RIGHT_EYEBROW,
            face_landmarks,
            width,
            height
        )

        eyebrow_mask = cv2.bitwise_or(
            left_eyebrow_mask,
            right_eyebrow_mask
        )

        kernel = np.ones((9,9), np.uint8)
        eyebrow_mask = cv2.dilate(
            eyebrow_mask,
            kernel,
            iterations=1
        )

        eyebrow_hair_mask = (
            self.get_eyebrow_hair_mask(
                image,
                eyebrow_mask
            )
        )

        # 피부 ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ

        skin_mask = face_mask.copy()

        skin_mask[left_eye_mask == 255] = 0
        skin_mask[right_eye_mask == 255] = 0

        skin_mask[lip_mask == 255] = 0

        skin_mask[eyebrow_mask == 255] = 0

        # skin_region = cv2.bitwise_and(
        #     image,
        #     image,
        #     mask = skin_mask
        # )

        # 홍채 ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ

        iris_mask = self.get_iris_mask(
            face_landmarks,
            width,
            height
        )

        # 이마 ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ

        forehead_mask = self.get_forehead_sample_mask(
            face_landmarks,
            width,
            height
        )

        forehead_mask = cv2.bitwise_and(
            forehead_mask,
            skin_mask
        )

        # 볼 ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ

        left_cheek_mask = self.get_left_cheek_mask(
            face_landmarks,
            width,
            height
        )

        left_cheek_mask = cv2.bitwise_and(
            left_cheek_mask,
            skin_mask
        )

        right_cheek_mask = self.get_right_cheek_mask(
            face_landmarks,
            width,
            height
        )

        right_cheek_mask = cv2.bitwise_and(
            right_cheek_mask,
            skin_mask
        )

        # 전체 region ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ

        skin_region = cv2.bitwise_and(
            image,
            image,
            mask = skin_mask
        )

        lip_region = cv2.bitwise_and(
            image,
            image,
            mask = lip_mask
        )

        iris_region = cv2.bitwise_and(
            image,
            image,
            mask = iris_mask
        )

        eyebrow_region = cv2.bitwise_and(
            image,
            image,
            mask = eyebrow_hair_mask
        )

        forehead_region = cv2.bitwise_and(
            image,
            image,
            mask = forehead_mask
        )

        left_cheek_region = cv2.bitwise_and(
            image,
            image,
            mask = left_cheek_mask
        )

        right_cheek_region = cv2.bitwise_and(
            image,
            image,
            mask = right_cheek_mask
        )

        return {

            # mask
            "skin_mask": skin_mask,
            "forehead_mask": forehead_mask,
            "left_cheek_mask":left_cheek_mask,
            "right_cheek_mask":right_cheek_mask,

            # regions
            "skin_region": skin_region,
            "lip_region": lip_region,
            "iris_region": iris_region,
            "eyebrow_region": eyebrow_region,
            "forehead_region": forehead_region,
            "left_cheek_region":left_cheek_region,
            "right_cheek_region": right_cheek_region
        }