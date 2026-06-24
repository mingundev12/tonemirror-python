import cv2
import numpy as np

from extractor.common import (
    FACE_OVAL,
    LEFT_EYE,
    RIGHT_EYE,
    LIPS,
    LEFT_EYEBROW,
    RIGHT_EYEBROW,
    create_mask
)


class SkinExtractor:

    # 눈썹 실제 털 영역 추출
    def get_eyebrow_hair_mask(self, image, eyebrow_mask):

        eyebrow_region = cv2.bitwise_and(
            image,
            image,
            mask=eyebrow_mask
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


    def extract(self, image, face_landmarks, width, height):

        # 얼굴
        face_mask = create_mask(
            FACE_OVAL,
            face_landmarks,
            width,
            height
        )

        # 눈
        left_eye_mask = create_mask(
            LEFT_EYE,
            face_landmarks,
            width,
            height
        )

        right_eye_mask = create_mask(
            RIGHT_EYE,
            face_landmarks,
            width,
            height
        )

        # 입술
        lip_mask = create_mask(
            LIPS,
            face_landmarks,
            width,
            height
        )

        # 눈썹
        left_eyebrow_mask = create_mask(
            LEFT_EYEBROW,
            face_landmarks,
            width,
            height
        )

        right_eyebrow_mask = create_mask(
            RIGHT_EYEBROW,
            face_landmarks,
            width,
            height
        )

        eyebrow_mask = cv2.bitwise_or(
            left_eyebrow_mask,
            right_eyebrow_mask
        )

        kernel = np.ones(
            (9, 9),
            np.uint8
        )

        eyebrow_mask = cv2.dilate(
            eyebrow_mask,
            kernel,
            iterations=1
        )

        eyebrow_hair_mask = self.get_eyebrow_hair_mask(
            image,
            eyebrow_mask
        )

        # 피부 마스크
        skin_mask = face_mask.copy()

        skin_mask[left_eye_mask == 255] = 0
        skin_mask[right_eye_mask == 255] = 0

        skin_mask[lip_mask == 255] = 0
        skin_mask[eyebrow_mask == 255] = 0


        # Region 생성
        skin_region = cv2.bitwise_and(
            image,
            image,
            mask=skin_mask
        )

        lip_region = cv2.bitwise_and(
            image,
            image,
            mask=lip_mask
        )

        eyebrow_region = cv2.bitwise_and(
            image,
            image,
            mask=eyebrow_hair_mask
        )

        return {
            "skin_mask": skin_mask,
            "skin_region": skin_region,
            "lip_region": lip_region,
            "eyebrow_region": eyebrow_region
        }