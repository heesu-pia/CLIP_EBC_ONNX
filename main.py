import argparse
import json
import os

from custom.clip_ebc_onnx import ClipEBCOnnx


def parse_args():
    parser = argparse.ArgumentParser(description="CLIP-EBC Crowd Counting (ONNX)")
    parser.add_argument(
        "--img-path", type=str, default=None, help="Path of an input image"
    )
    parser.add_argument(
        "--img-dir", type=str, default=None, help="Directory of input images"
    )
    parser.add_argument(
        "--model",
        default="assets/CLIP_EBC_nwpu_rmse_onnx.onnx",
        help="Path to ONNX model",
    )
    parser.add_argument(
        "--visualize",
        choices=["density", "dots", "all", "none"],
        default="none",
        help="Visualization type",
    )
    parser.add_argument(
        "--save", action="store_true", help="Save visualization results"
    )
    parser.add_argument(
        "--output-dir", default="results", help="Directory to save results"
    )

    # 시각화 관련 매개변수
    parser.add_argument(
        "--alpha", type=float, default=0.5, help="Alpha value for density map"
    )
    parser.add_argument(
        "--dot-size", type=int, default=20, help="Dot size for dot visualization"
    )
    parser.add_argument(
        "--sigma", type=float, default=1, help="Sigma value for Gaussian filter"
    )
    parser.add_argument(
        "--percentile",
        type=float,
        default=97,
        help="Percentile threshold for dot visualization",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    if (args.img_path is None and args.img_dir is None) or (
        args.img_path is not None and args.img_dir is not None
    ):
        raise ValueError("Exactly one of --img_path or --img_dir must be provided.")

    # 모델 초기화 - ONNX 버전
    model = ClipEBCOnnx(onnx_model_path=args.model)

    # 출력 디렉토리 생성
    if args.save:
        os.makedirs(args.output_dir, exist_ok=True)

    if args.img_path:
        # 예측 수행
        count = model.predict(args.image)
        print(f"예측된 군중 수: {count:.2f}")
        img_filename = os.path.splitext(os.path.basename(args.image))[0]
        result = {"img_id": img_filename, "human_num": float(count)}

        folder_name = args.image.split("/")[7]
        json_path = os.path.join(args.output_dir, folder_name, f"{img_filename}.json")
        os.makedirs(os.path.dirname(json_path), exist_ok=True)
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=4)
        print(f"Count result saved to: {json_path}")

        # 시각화
        if args.visualize in ["density", "all"]:
            save_path = (
                os.path.join(args.output_dir, "density_map.png") if args.save else None
            )
            fig, density_map = model.visualize_density_map(
                alpha=args.alpha, save=args.save, save_path=save_path
            )

        if args.visualize in ["dots", "all"]:
            save_path = (
                os.path.join(args.output_dir, "dot_map.png") if args.save else None
            )
            canvas, dot_map = model.visualize_dots(
                dot_size=args.dot_size,
                sigma=args.sigma,
                percentile=args.percentile,
                save=args.save,
                save_path=save_path,
            )

            # matplotlib figure 닫기 (메모리 누수 방지)
            if args.visualize in ["density", "all"]:
                import matplotlib.pyplot as plt

                plt.close(fig)

            if args.visualize in ["dots", "all"]:
                import matplotlib.pyplot as plt

                plt.close(canvas.figure)

    else:
        # img_dir이 주어진 경우
        supported_exts = (".jpg", ".jpeg", ".png")
        img_files = [
            f for f in os.listdir(args.img_dir) if f.lower().endswith(supported_exts)
        ]

        if not img_files:
            raise ValueError(f"No image files found in directory: {args.img_dir}")

        for f in img_files:
            img_path = os.path.join(args.img_dir, f)

            # 예측 수행
            count = model.predict(img_path)
            print(f"{f} → 예측된 군중 수: {count:.2f}")

            # 결과 JSON 저장
            img_filename = os.path.splitext(f)[0]
            result = {"img_id": f, "human_num": float(count)}

            # NOTE: Hard coded foler name
            folder_name = args.img_dir.split("/")[7]
            json_path = os.path.join(
                args.output_dir, folder_name, f"{img_filename}.json"
            )
            os.makedirs(os.path.dirname(json_path), exist_ok=True)
            with open(json_path, "w", encoding="utf-8") as jf:
                json.dump(result, jf, ensure_ascii=False, indent=4)
            print(f"Count result saved to: {json_path}")

            # 시각화
            if args.visualize in ["density", "all"]:
                save_path = (
                    os.path.join(
                        args.output_dir, folder_name, f"{img_filename}_density.png"
                    )
                    if args.save
                    else None
                )
                fig, density_map = model.visualize_density_map(
                    alpha=args.alpha, save=args.save, save_path=save_path
                )

            if args.visualize in ["dots", "all"]:
                save_path = (
                    os.path.join(
                        args.output_dir, folder_name, f"{img_filename}_dot.png"
                    )
                    if args.save
                    else None
                )
                canvas, dot_map = model.visualize_dots(
                    dot_size=args.dot_size,
                    sigma=args.sigma,
                    percentile=args.percentile,
                    save=args.save,
                    save_path=save_path,
                )

            # matplotlib figure 닫기
            if args.visualize in ["density", "all"]:
                import matplotlib.pyplot as plt

                plt.close(fig)
            if args.visualize in ["dots", "all"]:
                import matplotlib.pyplot as plt

                plt.close(canvas.figure)


if __name__ == "__main__":
    main()
