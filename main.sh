#!/bin/bash
# filepath: /home/hs.kim/Workspace/CLIP_EBC_ONNX/main.sh

# 입력 이미지 경로 리스트
images=(
    "test_images/101093_앞산빨래터공원 군중계수 테스트 요청 자료/vlcsnap-2025-07-11-15h00m00s564.png"
    "test_images/101093_앞산빨래터공원 군중계수 테스트 요청 자료/vlcsnap-2025-07-11-15h02m11s484.png"
    "test_images/101093_앞산빨래터공원 군중계수 테스트 요청 자료/vlcsnap-2025-07-11-15h02m28s914.png"
    "test_images/101093_앞산빨래터공원 군중계수 테스트 요청 자료/vlcsnap-2025-07-11-15h02m42s545.png"
    "test_images/101093_앞산빨래터공원 군중계수 테스트 요청 자료/vlcsnap-2025-07-11-15h03m04s666.png"
    "test_images/101093_앞산빨래터공원 군중계수 테스트 요청 자료/vlcsnap-2025-07-11-15h03m14s871.png"
    "test_images/101093_앞산빨래터공원 군중계수 테스트 요청 자료/vlcsnap-2025-07-11-15h03m20s852.png"
    "test_images/101093_앞산빨래터공원 군중계수 테스트 요청 자료/vlcsnap-2025-07-11-15h03m24s213.png"
    "test_images/101093_앞산빨래터공원 군중계수 테스트 요청 자료/vlcsnap-2025-07-11-15h03m31s750.png"
    "test_images/101093_앞산빨래터공원 군중계수 테스트 요청 자료/vlcsnap-2025-07-11-15h03m38s371.png"
    "test_images/101093_앞산빨래터공원 군중계수 테스트 요청 자료/vlcsnap-2025-07-11-15h03m47s878.png"
    "test_images/101093_앞산빨래터공원 군중계수 테스트 요청 자료/vlcsnap-2025-07-11-15h04m15s454.png"
    "test_images/101093_앞산빨래터공원 군중계수 테스트 요청 자료/vlcsnap-2025-07-11-15h05m12s689.png"
    "test_images/101095_앞산빨래터공원 군중계수 테스트 요청 자료/vlcsnap-2025-07-11-14h26m52s730.png"
    "test_images/101095_앞산빨래터공원 군중계수 테스트 요청 자료/vlcsnap-2025-07-11-14h28m52s238.png"
    "test_images/101095_앞산빨래터공원 군중계수 테스트 요청 자료/vlcsnap-2025-07-11-14h29m02s979.png"
    "test_images/101095_앞산빨래터공원 군중계수 테스트 요청 자료/vlcsnap-2025-07-11-14h29m12s654.png"
    "test_images/101095_앞산빨래터공원 군중계수 테스트 요청 자료/vlcsnap-2025-07-11-14h30m34s766.png"
    "test_images/101095_앞산빨래터공원 군중계수 테스트 요청 자료/vlcsnap-2025-07-11-14h32m35s809.png"
    "test_images/101095_앞산빨래터공원 군중계수 테스트 요청 자료/vlcsnap-2025-07-11-14h32m57s169.png"
    "test_images/101095_앞산빨래터공원 군중계수 테스트 요청 자료/vlcsnap-2025-07-11-14h33m30s378.png"
    "test_images/101095_앞산빨래터공원 군중계수 테스트 요청 자료/vlcsnap-2025-07-11-14h34m20s485.png"
    "test_images/101095_앞산빨래터공원 군중계수 테스트 요청 자료/vlcsnap-2025-07-11-14h39m25s404.png"
    "test_images/101095_앞산빨래터공원 군중계수 테스트 요청 자료/vlcsnap-2025-07-11-15h12m29s235.png"
    "test_images/101095_앞산빨래터공원 군중계수 테스트 요청 자료/vlcsnap-2025-07-11-15h12m33s097.png"
    "test_images/101095_앞산빨래터공원 군중계수 테스트 요청 자료/vlcsnap-2025-07-11-15h13m49s480.png"
    "test_images/101095_앞산빨래터공원 군중계수 테스트 요청 자료/vlcsnap-2025-07-11-15h13m57s881.png"
    "test_images/101095_앞산빨래터공원 군중계수 테스트 요청 자료/vlcsnap-2025-07-11-15h14m09s148.png"
    "test_images/101095_앞산빨래터공원 군중계수 테스트 요청 자료/vlcsnap-2025-07-11-15h14m18s166.png"
)

for img_path in "${images[@]}"; do
    # 출력 디렉토리 경로 생성
    out_dir="outputs/${img_path#test_images/}"
    out_dir="${out_dir%.png}"
    mkdir -p "$out_dir"

    python main.py \
        --image "$img_path" \
        --model 'clip_ebc_model.onnx' \
        --visualize 'density' \
        --save \
        --output-dir "$out_dir" \
        --alpha 0.5 \
        --dot-size 20 \
        --sigma 1 \
        --percentile 99.5
done
