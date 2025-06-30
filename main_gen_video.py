import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from custom.clip_ebc_onnx import ClipEBCOnnx


def process_video(
    input_video_path,
    output_density_path,
    output_dot_path,
    onnx_model_path="clip_ebc_model.onnx",
):
    model = ClipEBCOnnx(onnx_model_path=onnx_model_path)
    cap = cv2.VideoCapture(input_video_path)
    fps = 1  # 1fps로 저장
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    out_density = cv2.VideoWriter(output_density_path, fourcc, fps, (width, height))
    out_dot = cv2.VideoWriter(output_dot_path, fourcc, fps, (width, height))

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    orig_fps = cap.get(cv2.CAP_PROP_FPS)
    interval = int(orig_fps // fps) if orig_fps > 0 else 1

    frame_idx = 0
    pbar = tqdm(total=frame_count, desc="Processing video")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % interval == 0:
            count = model.predict(frame)

            # density map
            # fig, density_map = model.visualize_density_map(
            #     save_path="tmp_density_map.jpg", alpha=0.5, save=False
            # )

            # # fig가 Figure 객체가 아닐 경우, fig.figure로 Figure를 얻을 수 있음
            # if hasattr(fig, "figure"):
            #     plt.close(fig.figure)
            # else:
            #     plt.close(fig)

            # density_map = (
            #     np.uint8(density_map * 255)
            #     if density_map.max() <= 1.0
            #     else np.uint8(density_map)
            # )
            # density_map_bgr = (
            #     density_map
            #     if len(density_map.shape) == 3 and density_map.shape[2] == 3
            #     else cv2.cvtColor(density_map, cv2.COLOR_GRAY2BGR)
            # )
            # # 원본 프레임 크기에 맞게 padding 또는 crop 없이 비율 유지 리사이즈
            # h_src, w_src = density_map_bgr.shape[:2]
            # aspect_src = w_src / h_src
            # aspect_dst = width / height
            # if aspect_src > aspect_dst:
            #     # 좌우 기준으로 맞추고 위아래 패딩
            #     new_w = width
            #     new_h = int(width / aspect_src)
            #     resized = cv2.resize(density_map_bgr, (new_w, new_h))
            #     pad_top = (height - new_h) // 2
            #     pad_bottom = height - new_h - pad_top
            #     density_map_bgr = cv2.copyMakeBorder(
            #         resized,
            #         pad_top,
            #         pad_bottom,
            #         0,
            #         0,
            #         cv2.BORDER_CONSTANT,
            #         value=(0, 0, 0),
            #     )
            # else:
            #     # 위아래 기준으로 맞추고 좌우 패딩
            #     new_h = height
            #     new_w = int(height * aspect_src)
            #     resized = cv2.resize(density_map_bgr, (new_w, new_h))
            #     pad_left = (width - new_w) // 2
            #     pad_right = width - new_w - pad_left
            #     density_map_bgr = cv2.copyMakeBorder(
            #         resized,
            #         0,
            #         0,
            #         pad_left,
            #         pad_right,
            #         cv2.BORDER_CONSTANT,
            #         value=(0, 0, 0),
            #     )

            # out_density.write(density_map_bgr)

            # dot map
            fig, dot_map = model.visualize_dots(
                save_path="tmp_dot.jpg",
                dot_size=20,
                sigma=1,
                percentile=99.5,
                save=False,
            )
            # fig가 Figure 객체가 아닐 경우, fig.figure로 Figure를 얻을 수 있음
            if hasattr(fig, "figure"):
                plt.close(fig.figure)
            else:
                plt.close(fig)
            dot_map = (
                np.uint8(dot_map * 255) if dot_map.max() <= 1.0 else np.uint8(dot_map)
            )
            dot_map_bgr = (
                dot_map
                if len(dot_map.shape) == 3 and dot_map.shape[2] == 3
                else cv2.cvtColor(dot_map, cv2.COLOR_GRAY2BGR)
            )
            # 원본 프레임 크기에 맞게 padding 또는 crop 없이 비율 유지 리사이즈
            h_src, w_src = dot_map_bgr.shape[:2]
            aspect_src = w_src / h_src
            aspect_dst = width / height
            if aspect_src > aspect_dst:
                new_w = width
                new_h = int(width / aspect_src)
                resized = cv2.resize(dot_map_bgr, (new_w, new_h))
                pad_top = (height - new_h) // 2
                pad_bottom = height - new_h - pad_top
                dot_map_bgr = cv2.copyMakeBorder(
                    resized,
                    pad_top,
                    pad_bottom,
                    0,
                    0,
                    cv2.BORDER_CONSTANT,
                    value=(0, 0, 0),
                )
            else:
                new_h = height
                new_w = int(height * aspect_src)
                resized = cv2.resize(dot_map_bgr, (new_w, new_h))
                pad_left = (width - new_w) // 2
                pad_right = width - new_w - pad_left
                dot_map_bgr = cv2.copyMakeBorder(
                    resized,
                    0,
                    0,
                    pad_left,
                    pad_right,
                    cv2.BORDER_CONSTANT,
                    value=(0, 0, 0),
                )

            out_dot.write(dot_map_bgr)

        frame_idx += 1
        pbar.update(1)
    pbar.close()
    cap.release()
    out_density.release()
    out_dot.release()
    print("영상 저장 완료:", output_density_path, output_dot_path)


def main():
    os.makedirs("outputs", exist_ok=True)

    video_list = [
        # "assets/ai허브_군중밀집_1.mp4",
        # "assets/ai허브_군중밀집_2.mp4",
        # "assets/ai허브_군중밀집_3.mp4",
        # "assets/ai허브_군중밀집_4.mp4",
        # "assets/대기열2.mp4",
        # "assets/대기열3.mp4",
        # "assets/대기열4.mp4",
        "assets/clip_REC-0002-A.mp4",
        "assets/clip_REC-0004-A.mp4",
        "assets/clip_REC-0005-A-2.mp4",
        "assets/clip_REC-0006-A.mp4",
        "assets/clip_REC-0009-A.mp4",
    ]

    for input_video_path in video_list:
        base = os.path.splitext(os.path.basename(input_video_path))[0]
        ext = os.path.splitext(input_video_path)[1]
        output_density_path = f"outputs/{base}_density_map{ext}"
        output_dot_path = f"outputs/{base}_dot_map{ext}"
        process_video(input_video_path, output_density_path, output_dot_path)


if __name__ == "__main__":
    main()
