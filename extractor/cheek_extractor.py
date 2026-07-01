import cv2
import numpy as np

class CheekExtractor:

    def get_left_cheek_mask(
        self,
        face_landmarks,
        width,
        height
    ):

        cheek = face_landmarks.landmark[205]

        cx = int(cheek.x * width)
        cy = int(cheek.y * height)

        face_left = face_landmarks.landmark[234]
        face_right = face_landmarks.landmark[454]

        face_width = int(
            abs(face_right.x - face_left.x) * width
        )

        mask = np.zeros(
            (height, width),
            dtype=np.uint8
        )

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


    def get_right_cheek_mask(
        self,
        face_landmarks,
        width,
        height
    ):

        cheek = face_landmarks.landmark[425]

        cx = int(cheek.x * width)
        cy = int(cheek.y * height)

        face_left = face_landmarks.landmark[234]
        face_right = face_landmarks.landmark[454]

        face_width = int(
            abs(face_right.x - face_left.x) * width
        )

        mask = np.zeros(
            (height, width),
            dtype=np.uint8
        )

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


    def extract(
        self,
        image,
        face_landmarks,
        width,
        height,
        skin_mask
    ):

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


        left_cheek_region = cv2.bitwise_and(
            image,
            image,
            mask=left_cheek_mask
        )

        right_cheek_region = cv2.bitwise_and(
            image,
            image,
            mask=right_cheek_mask
        )


        return {
            "left_cheek_region": left_cheek_region,
            "right_cheek_region": right_cheek_region
        }