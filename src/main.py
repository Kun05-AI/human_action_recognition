from .config import INPUT_DIR, OUTPUT_DIR
from .video_processor import VideoProcessor


def main() -> None:
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    video_files = sorted(
        {
            path.resolve()
            for path in INPUT_DIR.iterdir()
            if path.is_file()
            and path.suffix.lower() == ".mp4"
        }
    )

    if not video_files:
        raise FileNotFoundError(
            f"No MP4 files found in: {INPUT_DIR}"
        )

    processor = VideoProcessor()

    print(f"Found {len(video_files)} video(s).")

    for input_path in video_files:

        output_path = (
            OUTPUT_DIR
            / f"{input_path.stem}_phase2.mp4"
        )

        print()
        print(
            f"Processing: {input_path.name}"
        )
        print(
            f"Output:    {output_path.name}"
        )

        processor.process(
            input_path=input_path,
            output_path=output_path,
        )

        print(
            f"Finished:  {input_path.name}"
        )

    print()
    print(
        "All videos processed successfully."
    )


if __name__ == "__main__":
    main()