class VideoProcessor:
    def __init__(self) -> None:
        self.detector = PoseDetector()
        self.identity_manager = IdentityManager()

        self.pose_buffer_manager = PoseBufferManager(
            max_length=POSE_BUFFER_SIZE
        )