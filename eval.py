import json
import os
from glob import glob

from natsort import natsorted
from rich.progress import track


def main():
    folder_name_list = os.listdir(
        "/home/hs.kim/Workspace/CLIP_EBC_ONNX/dataset/incheon_labeling_data_original"
    )

    folder_name_list = [
        s for s in folder_name_list if isinstance(s, str) and s.startswith("TEST")
    ]
    folder_name_list = natsorted(folder_name_list)

    gt_dirs = [
        f"/home/hs.kim/Workspace/CLIP_EBC_ONNX/dataset/incheon_labeling_data_original/{folder_name}/json"
        for folder_name in folder_name_list
    ]
    pred_dirs = [
        f"/home/hs.kim/Workspace/CLIP_EBC_ONNX/output_IAR/{folder_name}"
        for folder_name in folder_name_list
    ]

    # acc_list = []

    running_sum_ratio = 0.0  # per-image ratio 누적합
    n_samples = 0
    sum_clamped_pred = 0.0  # 글로벌 비율 분자: sum(min(pred, gt))
    sum_gt = 0.0

    for gt_dir, pred_dir in track(zip(gt_dirs, pred_dirs), total=len(gt_dirs)):

        json_list = os.listdir(gt_dir)
        json_list = natsorted([j for j in json_list if j.endswith("json")])

        for json_filename in json_list:
            gt_json_path = os.path.join(gt_dir, json_filename)
            pred_json_path = os.path.join(pred_dir, json_filename)

            with open(gt_json_path, "r") as f:
                gt_json_data = json.load(f)
            with open(pred_json_path, "r") as f:
                pred_json_data = json.load(f)

            gt_people_count = gt_json_data["human_num"]
            pred_people_count = pred_json_data["human_num"]

            # acc = (
            #     pred_people_count / gt_people_count
            #     if pred_people_count <= gt_people_count
            #     else 1.0
            # )

            # acc_list.append(acc)
            # print(f"File: {pred_json_path} | People counting rate: {acc*100} % | Accumulated people counting rate: ")

            ratio = (
                min(pred_people_count / gt_people_count, 1.0)
                if gt_people_count != 0
                else 1.0
            )

            n_samples += 1
            running_sum_ratio += ratio

            sum_clamped_pred += min(pred_people_count, gt_people_count)
            sum_gt += gt_people_count

            avg_ratio = running_sum_ratio / n_samples
            global_ratio = (sum_clamped_pred / sum_gt) if sum_gt > 0 else 0.0

            print(
                f"File: {pred_json_path} | "
                f"ratio: {ratio*100:.2f}% | "
                f"Accumulated(avg): {avg_ratio*100:.2f}% | "
                f"Accumulated(global): {global_ratio*100:.2f}%"
            )


if __name__ == "__main__":
    main()
