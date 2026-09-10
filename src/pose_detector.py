from ultralytics import YOLO

from .config import (
    MODEL_PATH,
    DEVICE,
    IMAGE_SIZE,
    CONF_THRESHOLD,
    IOU_THRESHOLD,
    PERSON_CLASS_ID,
)


class PoseDetector:
    def __init__(self) -> None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Model not found: {MODEL_PATH}"
            )

        self.model = YOLO(str(MODEL_PATH))

    def track(self, frame):
        """
        Run YOLO26-Pose + ByteTrack on one frame.

        persist=True keeps the tracker state between frames.
        """

        results = self.model.track(
            source=frame,
            persist=True,
            tracker="bytetrack.yaml",
            device=DEVICE,
            imgsz=IMAGE_SIZE,
            conf=CONF_THRESHOLD,
            iou=IOU_THRESHOLD,
            classes=[PERSON_CLASS_ID],
            verbose=False,
        )

        return results[0]