import argparse

from demo_runtime import normalized_input_path, run_capture


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to input video file.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_capture(normalized_input_path(args.input), "Video Deepfake Risk Prototype")

