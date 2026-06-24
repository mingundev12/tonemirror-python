import cv2
import numpy as np

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

# 랜드마크 -> Polygon Mask 생성
def create_mask(
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