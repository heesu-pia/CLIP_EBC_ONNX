#!/bin/bash
# filepath: /home/hs.kim/Workspace/CLIP_EBC_ONNX/main.sh

# Daegu
# 입력 이미지 경로 리스트
# images=(
#     "test_images/101093_앞산빨래터공원 군중계수 테스트 요청 자료/vlcsnap-2025-07-11-15h00m00s564.png"
#     "test_images/101093_앞산빨래터공원 군중계수 테스트 요청 자료/vlcsnap-2025-07-11-15h02m11s484.png"
#     "test_images/101093_앞산빨래터공원 군중계수 테스트 요청 자료/vlcsnap-2025-07-11-15h02m28s914.png"
#     "test_images/101093_앞산빨래터공원 군중계수 테스트 요청 자료/vlcsnap-2025-07-11-15h02m42s545.png"
#     "test_images/101093_앞산빨래터공원 군중계수 테스트 요청 자료/vlcsnap-2025-07-11-15h03m04s666.png"
#     "test_images/101093_앞산빨래터공원 군중계수 테스트 요청 자료/vlcsnap-2025-07-11-15h03m14s871.png"
#     "test_images/101093_앞산빨래터공원 군중계수 테스트 요청 자료/vlcsnap-2025-07-11-15h03m20s852.png"
#     "test_images/101093_앞산빨래터공원 군중계수 테스트 요청 자료/vlcsnap-2025-07-11-15h03m24s213.png"
#     "test_images/101093_앞산빨래터공원 군중계수 테스트 요청 자료/vlcsnap-2025-07-11-15h03m31s750.png"
#     "test_images/101093_앞산빨래터공원 군중계수 테스트 요청 자료/vlcsnap-2025-07-11-15h03m38s371.png"
#     "test_images/101093_앞산빨래터공원 군중계수 테스트 요청 자료/vlcsnap-2025-07-11-15h03m47s878.png"
#     "test_images/101093_앞산빨래터공원 군중계수 테스트 요청 자료/vlcsnap-2025-07-11-15h04m15s454.png"
#     "test_images/101093_앞산빨래터공원 군중계수 테스트 요청 자료/vlcsnap-2025-07-11-15h05m12s689.png"
#     "test_images/101095_앞산빨래터공원 군중계수 테스트 요청 자료/vlcsnap-2025-07-11-14h26m52s730.png"
#     "test_images/101095_앞산빨래터공원 군중계수 테스트 요청 자료/vlcsnap-2025-07-11-14h28m52s238.png"
#     "test_images/101095_앞산빨래터공원 군중계수 테스트 요청 자료/vlcsnap-2025-07-11-14h29m02s979.png"
#     "test_images/101095_앞산빨래터공원 군중계수 테스트 요청 자료/vlcsnap-2025-07-11-14h29m12s654.png"
#     "test_images/101095_앞산빨래터공원 군중계수 테스트 요청 자료/vlcsnap-2025-07-11-14h30m34s766.png"
#     "test_images/101095_앞산빨래터공원 군중계수 테스트 요청 자료/vlcsnap-2025-07-11-14h32m35s809.png"
#     "test_images/101095_앞산빨래터공원 군중계수 테스트 요청 자료/vlcsnap-2025-07-11-14h32m57s169.png"
#     "test_images/101095_앞산빨래터공원 군중계수 테스트 요청 자료/vlcsnap-2025-07-11-14h33m30s378.png"
#     "test_images/101095_앞산빨래터공원 군중계수 테스트 요청 자료/vlcsnap-2025-07-11-14h34m20s485.png"
#     "test_images/101095_앞산빨래터공원 군중계수 테스트 요청 자료/vlcsnap-2025-07-11-14h39m25s404.png"
#     "test_images/101095_앞산빨래터공원 군중계수 테스트 요청 자료/vlcsnap-2025-07-11-15h12m29s235.png"
#     "test_images/101095_앞산빨래터공원 군중계수 테스트 요청 자료/vlcsnap-2025-07-11-15h12m33s097.png"
#     "test_images/101095_앞산빨래터공원 군중계수 테스트 요청 자료/vlcsnap-2025-07-11-15h13m49s480.png"
#     "test_images/101095_앞산빨래터공원 군중계수 테스트 요청 자료/vlcsnap-2025-07-11-15h13m57s881.png"
#     "test_images/101095_앞산빨래터공원 군중계수 테스트 요청 자료/vlcsnap-2025-07-11-15h14m09s148.png"
#     "test_images/101095_앞산빨래터공원 군중계수 테스트 요청 자료/vlcsnap-2025-07-11-15h14m18s166.png"
# )

# for img_path in "${images[@]}"; do
#     # 출력 디렉토리 경로 생성
#     out_dir="outputs/${img_path#test_images/}"
#     out_dir="${out_dir%.png}"
#     mkdir -p "$out_dir"

#     python main.py \
#         --image "$img_path" \
#         --model 'clip_ebc_model.onnx' \
#         --visualize 'density' \
#         --save \
#         --output-dir "$out_dir" \
#         --alpha 0.5 \
#         --dot-size 20 \
#         --sigma 1 \
#         --percentile 99.5
# done

# IAR
set -euo pipefail

# 입력 이미지 경로 리스트
image_dirs=(
    /home/hs.kim/Workspace/CLIP_EBC_ONNX/dataset/incheon_labeling_data_original/TEST001/images/
    # /home/hs.kim/Workspace/CLIP_EBC_ONNX/dataset/incheon_labeling_data_original/TEST001-2/images/
    # /home/hs.kim/Workspace/CLIP_EBC_ONNX/dataset/incheon_labeling_data_original/TEST001-3/images/
    # /home/hs.kim/Workspace/CLIP_EBC_ONNX/dataset/incheon_labeling_data_original/TEST001-4/images/
    # /home/hs.kim/Workspace/CLIP_EBC_ONNX/dataset/incheon_labeling_data_original/TEST001-5/images/
    # /home/hs.kim/Workspace/CLIP_EBC_ONNX/dataset/incheon_labeling_data_original/TEST001-6/images/
    # /home/hs.kim/Workspace/CLIP_EBC_ONNX/dataset/incheon_labeling_data_original/TEST002/images/
    # /home/hs.kim/Workspace/CLIP_EBC_ONNX/dataset/incheon_labeling_data_original/TEST002-2/images/
    # /home/hs.kim/Workspace/CLIP_EBC_ONNX/dataset/incheon_labeling_data_original/TEST002-3/images/
    # /home/hs.kim/Workspace/CLIP_EBC_ONNX/dataset/incheon_labeling_data_original/TEST002-4/images/
    # /home/hs.kim/Workspace/CLIP_EBC_ONNX/dataset/incheon_labeling_data_original/TEST002-5/images/
    # /home/hs.kim/Workspace/CLIP_EBC_ONNX/dataset/incheon_labeling_data_original/TEST002-6/images/
    # /home/hs.kim/Workspace/CLIP_EBC_ONNX/dataset/incheon_labeling_data_original/TEST003/images/
    # /home/hs.kim/Workspace/CLIP_EBC_ONNX/dataset/incheon_labeling_data_original/TEST003-2/images/
    # /home/hs.kim/Workspace/CLIP_EBC_ONNX/dataset/incheon_labeling_data_original/TEST003-3/images/
    # /home/hs.kim/Workspace/CLIP_EBC_ONNX/dataset/incheon_labeling_data_original/TEST003-4/images/
    # /home/hs.kim/Workspace/CLIP_EBC_ONNX/dataset/incheon_labeling_data_original/TEST003-6/images/
    # /home/hs.kim/Workspace/CLIP_EBC_ONNX/dataset/incheon_labeling_data_original/TEST004-2-no-lines/images/
    # /home/hs.kim/Workspace/CLIP_EBC_ONNX/dataset/incheon_labeling_data_original/TEST004-3-no-lines/images/
    # /home/hs.kim/Workspace/CLIP_EBC_ONNX/dataset/incheon_labeling_data_original/TEST004-4-no-lines/images/
    # /home/hs.kim/Workspace/CLIP_EBC_ONNX/dataset/incheon_labeling_data_original/TEST004-6-no-lines/images/
    # /home/hs.kim/Workspace/CLIP_EBC_ONNX/dataset/incheon_labeling_data_original/TEST004-no-lines/images/
    # /home/hs.kim/Workspace/CLIP_EBC_ONNX/dataset/incheon_labeling_data_original/TEST005-2/images/
    # /home/hs.kim/Workspace/CLIP_EBC_ONNX/dataset/incheon_labeling_data_original/TEST005-3/images/
    # /home/hs.kim/Workspace/CLIP_EBC_ONNX/dataset/incheon_labeling_data_original/TEST005-4/images/
    # /home/hs.kim/Workspace/CLIP_EBC_ONNX/dataset/incheon_labeling_data_original/TEST005-5/images/
    # /home/hs.kim/Workspace/CLIP_EBC_ONNX/dataset/incheon_labeling_data_original/TEST006/images/
    # /home/hs.kim/Workspace/CLIP_EBC_ONNX/dataset/incheon_labeling_data_original/TEST006-2/images/
    # /home/hs.kim/Workspace/CLIP_EBC_ONNX/dataset/incheon_labeling_data_original/TEST006-4/images/
    # /home/hs.kim/Workspace/CLIP_EBC_ONNX/dataset/incheon_labeling_data_original/TEST007/images/
    # /home/hs.kim/Workspace/CLIP_EBC_ONNX/dataset/incheon_labeling_data_original/TEST007-2/images/
    # /home/hs.kim/Workspace/CLIP_EBC_ONNX/dataset/incheon_labeling_data_original/TEST007-3/images/
    # /home/hs.kim/Workspace/CLIP_EBC_ONNX/dataset/incheon_labeling_data_original/TEST007-4/images/
    # /home/hs.kim/Workspace/CLIP_EBC_ONNX/dataset/incheon_labeling_data_original/TEST007-5/images/
    # /home/hs.kim/Workspace/CLIP_EBC_ONNX/dataset/incheon_labeling_data_original/TEST007-6/images/
    # /home/hs.kim/Workspace/CLIP_EBC_ONNX/dataset/incheon_labeling_data_original/TEST008/images/
    # /home/hs.kim/Workspace/CLIP_EBC_ONNX/dataset/incheon_labeling_data_original/TEST008-2/images/
    # /home/hs.kim/Workspace/CLIP_EBC_ONNX/dataset/incheon_labeling_data_original/TEST008-3/images/
    # /home/hs.kim/Workspace/CLIP_EBC_ONNX/dataset/incheon_labeling_data_original/TEST008-5/images/
    # /home/hs.kim/Workspace/CLIP_EBC_ONNX/dataset/incheon_labeling_data_original/TEST008-6/images/
    # /home/hs.kim/Workspace/CLIP_EBC_ONNX/dataset/incheon_labeling_data_original/TEST009/images/
    # /home/hs.kim/Workspace/CLIP_EBC_ONNX/dataset/incheon_labeling_data_original/TEST010/images/
    # /home/hs.kim/Workspace/CLIP_EBC_ONNX/dataset/incheon_labeling_data_original/TEST010-2/images/
    # /home/hs.kim/Workspace/CLIP_EBC_ONNX/dataset/incheon_labeling_data_original/TEST010-3/images/
    # /home/hs.kim/Workspace/CLIP_EBC_ONNX/dataset/incheon_labeling_data_original/TEST010-4/images/
    # /home/hs.kim/Workspace/CLIP_EBC_ONNX/dataset/incheon_labeling_data_original/TEST010-5/images/
)

# python main.py \
#     --img-dir "/home/hs.kim/Workspace/CLIP_EBC_ONNX/dataset/incheon_labeling_data_original/TEST001/images/" \
#     --model 'clip_ebc_model.onnx' \
#     --save \
#     --output-dir 'output_IAR' \
#     --alpha 0.5 \
#     --dot-size 20 \
#     --sigma 1 \
#     --percentile 99.5


# 공통 옵션 묶어두기
COMMON_ARGS=(
  --model 'clip_ebc_model.onnx'
  --save
  --output-dir 'output_IAR'
  --visualize 'all'
  --alpha 0.5
  --dot-size 20
  --sigma 1
  --percentile 99.5
)

# for dir in "${image_dirs[@]}"; do
#   echo ">>> Processing dir: $dir"
#   if [ -d "$dir" ]; then
#     python main.py --img-dir "$dir" "${COMMON_ARGS[@]}"
#   else
#     echo "!!! Skip: not found -> $dir" >&2
#   fi
# done

# 3) 로그 디렉토리 준비
LOG_DIR="output_IAR/logs"
mkdir -p "$LOG_DIR"

# 4) 실행
for dir in "${image_dirs[@]}"; do
  echo ">>> Processing dir: $dir"
  if [ -d "$dir" ]; then
    # '.../TEST001/images/' -> TEST001 같이 상위 폴더 이름 추출
    bn="$(basename "$(dirname "$dir")")"
    # 디렉토리별 로그 파일 (타임스탬프 포함)
    ts="$(date +%Y%m%d_%H%M%S)"
    log_file="$LOG_DIR/${bn}_${ts}.log"

    # 콘솔에도 출력 + 파일에도 저장
    python main.py --img-dir "$dir" "${COMMON_ARGS[@]}" 2>&1 | tee -a "$log_file"
  else
    echo "!!! Skip: not found -> $dir" >&2
  fi
done