import cv2
import numpy as np

from extractor.common import (
    LEFT_IRIS,
    RIGHT_IRIS
)


class IrisExtractor:

    def extract(self, image, face_landmarks, width, height):

        iris_mask = np.zeros(
            (height, width),
            dtype=np.uint8
        )

        for iris_points in [LEFT_IRIS, RIGHT_IRIS]:

            points = []

            for idx in iris_points:

                landmark = face_landmarks.landmark[idx]

                x = int(landmark.x * width)
                y = int(landmark.y * height)

                points.append((x, y))

            center_x = int(
                np.mean([p[0] for p in points])
            )

            center_y = int(
                np.mean([p[1] for p in points])
            )

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

        iris_region = cv2.bitwise_and(
            image,
            image,
            mask=iris_mask
        )

        return {
            "iris_region": iris_region
        }