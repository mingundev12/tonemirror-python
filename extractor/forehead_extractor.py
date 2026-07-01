import cv2
import numpy as np

from extractor.common import (
    LEFT_EYEBROW,
    RIGHT_EYEBROW
)


class ForeheadExtractor:

    def extract(
        self,
        image,
        face_landmarks,
        width,
        height,
        skin_mask
    ):

        # 얼굴 중앙
        center_x = int(
            face_landmarks.landmark[10].x * width
        )

        top_y = int(
            face_landmarks.landmark[10].y * height
        )

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

        forehead_mask = np.zeros(
            (height, width),
            dtype=np.uint8
        )

        cv2.rectangle(
            forehead_mask,
            (left_x, sample_top),
            (right_x, sample_bottom),
            255,
            -1
        )

        forehead_mask = cv2.bitwise_and(
            forehead_mask,
            skin_mask
        )

        forehead_region = cv2.bitwise_and(
            image,
            image,
            mask=forehead_mask
        )

        return {
            "forehead_region": forehead_region
        }