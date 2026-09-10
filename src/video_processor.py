from pathlib import Path

import cv2

from .config import (
    BOX_THICKNESS,
    POSE_BUFFER_SIZE,
)

from .identity import IdentityManager
from .pose_detector import PoseDetector
from .pose_buffer import PoseBufferManager


class VideoProcessor:
    def __init__(self) -> None:
        self.detector = PoseDetector()
        self.identity_manager = IdentityManager()

        self.pose_buffer_manager = PoseBufferManager(
            max_length=POSE_BUFFER_SIZE
    )

    def process(
        self,
        input_path: Path,
        output_path: Path,
    ) -> None:

        cap = cv2.VideoCapture(str(input_path))

        if not cap.isOpened():
            raise RuntimeError(
                f"Cannot open video: {input_path}"
            )

        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if fps <= 0:
            fps = 30.0

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")

        writer = cv2.VideoWriter(
            str(output_path),
            fourcc,
            fps,
            (width, height),
        )

        if not writer.isOpened():
            cap.release()
            raise RuntimeError(
                f"Cannot create output video: {output_path}"
            )

        frame_index = 0

        print(f"FPS: {fps:.2f}")
        print(f"Resolution: {width}x{height}")
        print(f"Frames: {total_frames}")

        while True:
            ret, frame = cap.read()

            if not ret:
                break

            frame_index += 1

            result = self.detector.track(frame)

            annotated = self._draw_tracking_result(
                frame,
                result,
                frame_index,
            )

            writer.write(annotated)

            if frame_index % 30 == 0:
                if total_frames > 0:
                    progress = (
                        frame_index / total_frames
                    ) * 100

                    print(
                        f"\rProgress: "
                        f"{progress:6.2f}%"
                        f" | Frame {frame_index}/{total_frames}",
                        end="",
                    )
                else:
                    print(
                        f"\rProcessed frame: {frame_index}",
                        end="",
                    )

        print()

        cap.release()
        writer.release()
        # =========================
        # Print pose buffer status
        # =========================

        print("\nPose buffers:")

        for tracker_id, buffer in (
            self.pose_buffer_manager.buffers.items()
        ):
            sequence = buffer.get_sequence()

            print(
                f"Tracker ID {tracker_id}: "
                f"frames={len(buffer)}, "
                f"shape={sequence.shape}, "
                f"ready={buffer.is_ready()}"
        )       

        print(f"Saved: {output_path}")

        print(f"Saved: {output_path}")

    def _draw_tracking_result(self, frame, result, frame_index):
        annotated = frame.copy()

        boxes = result.boxes
        keypoints = result.keypoints

        if boxes is None or len(boxes) == 0:
            return annotated

        if not boxes.is_track:
            return annotated

        if keypoints is None:
            return annotated

        tracker_ids = (
            boxes.id
            .int()
            .cpu()
            .tolist()
        )

        xyxy = (
            boxes.xyxy
            .int()
            .cpu()
            .tolist()
        )

        keypoint_data = keypoints.data.cpu().numpy()

        for index, (box, tracker_id) in enumerate(
            zip(xyxy, tracker_ids)
        ):
            x1, y1, x2, y2 = box

            identity = (
                self.identity_manager
                .get_identity(tracker_id)
            )

            # -------------------------------------------------
            # Pose buffer
            # -------------------------------------------------

            person_keypoints = keypoint_data[index]

            self.pose_buffer_manager.update(
                tracker_id=tracker_id,
                frame_index=frame_index,
                keypoints=person_keypoints,
            )

            # -------------------------------------------------
            # Bounding box
            # -------------------------------------------------

            cv2.rectangle(
                annotated,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                BOX_THICKNESS,
            )

            label = (
                f"ID: {identity.display_id} "
                f"| {identity.name}"
            )

            (text_width, text_height), baseline = (
                cv2.getTextSize(
                    label,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    2,
                )
            )

            text_y = max(
                y1,
                text_height + baseline + 5,
            )

            cv2.rectangle(
                annotated,
                (
                    x1,
                    text_y - text_height - baseline - 5,
                ),
                (
                    x1 + text_width + 8,
                    text_y,
                ),
                (0, 255, 0),
                -1,
            )

            cv2.putText(
                annotated,
                label,
                (x1 + 4, text_y - 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 0),
                2,
                cv2.LINE_AA,
            )

        annotated = result.plot(
            img=annotated,
            boxes=False,
            labels=False,
        )

        return annotated
    