from ultralytics import YOLO


class AccessoryValidator:

    def __init__(self):
        self.model = YOLO("YOLO/best.pt")

    def validate(self, image):

        results = self.model(image)

        boxes = results[0].boxes

        if len(boxes) == 0:
            return {
                "passed": True,
                "message": "정상"
            }

        for box in boxes:

            confidence = float(box.conf[0])

            if confidence < 0.5:
                continue

            cls_id = int(box.cls[0])
            name = self.model.names[cls_id]

            if name == "With_Glass":
                return {
                    "passed": False,
                    "message": "안경을 벗고 촬영해주세요."
                }

            if name == "With_Mask":
                return {
                    "passed": False,
                    "message": "마스크를 벗고 촬영해주세요."
                }

        return {
            "passed": True,
            "message": "정상"
        }