from collections import deque
from dataclasses import dataclass

import numpy as np


@dataclass
class PoseFrame:
    frame_index: int
    keypoints: np.ndarray


class PoseBuffer:
    """
    Stores a temporal sequence of pose keypoints for one tracked person.

    Each keypoint frame has shape:
        (17, 3)

    where the 3 values are:
        x, y, confidence
    """

    def __init__(self, max_length: int = 30) -> None:
        if max_length <= 0:
            raise ValueError("max_length must be > 0")

        self.max_length = max_length
        self.frames: deque[PoseFrame] = deque(
            maxlen=max_length
        )

    def add(
        self,
        frame_index: int,
        keypoints: np.ndarray,
    ) -> None:
        keypoints = np.asarray(
            keypoints,
            dtype=np.float32,
        )

        if keypoints.shape != (17, 3):
            raise ValueError(
                f"Expected keypoints shape (17, 3), "
                f"got {keypoints.shape}"
            )

        self.frames.append(
            PoseFrame(
                frame_index=frame_index,
                keypoints=keypoints.copy(),
            )
        )

    def is_ready(self) -> bool:
        return len(self.frames) == self.max_length

    def __len__(self) -> int:
        return len(self.frames)

    def get_sequence(self) -> np.ndarray:
        """
        Returns:
            shape = (T, 17, 3)
        """
        if not self.frames:
            return np.empty(
                (0, 17, 3),
                dtype=np.float32,
            )

        return np.stack(
            [item.keypoints for item in self.frames],
            axis=0,
        )

    def clear(self) -> None:
        self.frames.clear()


class PoseBufferManager:
    """
    Maintains one PoseBuffer per tracker ID.
    """

    def __init__(self, max_length: int = 30) -> None:
        self.max_length = max_length
        self.buffers: dict[int, PoseBuffer] = {}

    def update(
        self,
        tracker_id: int,
        frame_index: int,
        keypoints: np.ndarray,
    ) -> PoseBuffer:

        if tracker_id not in self.buffers:
            self.buffers[tracker_id] = PoseBuffer(
                max_length=self.max_length
            )

        buffer = self.buffers[tracker_id]

        buffer.add(
            frame_index=frame_index,
            keypoints=keypoints,
        )

        return buffer

    def get(
        self,
        tracker_id: int,
    ) -> PoseBuffer | None:

        return self.buffers.get(tracker_id)

    def remove(self, tracker_id: int) -> None:
        self.buffers.pop(
            tracker_id,
            None,
        )

    def clear(self) -> None:
        self.buffers.clear()