import os
import pandas as pd
import math
from tqdm import tqdm

def generate_frame_labels(input_csv_path, output_csv_path, fps=30):
    """
    Chuyển đổi nhãn thời gian (global timestamp) từ file audio_ball_hits.csv
    thành danh sách Frame Index toàn cục.
    """
    
    if not os.path.exists(input_csv_path):
        print(f"Lỗi: Không tìm thấy file {input_csv_path}")
        return

    print(f"Đang đọc file: {input_csv_path}")
    df = pd.read_csv(input_csv_path)
    
    if df.empty:
        print("File CSV rỗng.")
        return

    if 'start' not in df.columns or 'end' not in df.columns:
        print("Lỗi: File CSV thiếu cột 'start' hoặc 'end'.")
        return

    print(f"Đang xử lý {len(df)} dòng dữ liệu với FPS={fps}...")
    
    hit_frames_global = []

    # Duyệt qua từng dòng hit
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Converting"):
        t_start = row['start']
        t_end = row['end']
        
        f_start = int(math.floor(t_start * fps))
        f_end = int(math.floor(t_end * fps))
        
        for f_idx in range(f_start, f_end + 1):
            hit_frames_global.append(f_idx)

    hit_frames_global = sorted(list(set(hit_frames_global)))
    
    out_df = pd.DataFrame({
        'frame_idx': hit_frames_global,
        'label': 1
    })
    
    out_df.to_csv(output_csv_path, index=False)
    
    print(f"HOÀN TẤT! Tìm thấy {len(hit_frames_global)} frames có tiếng bóng.")
    print(f"Kết quả lưu tại: {output_csv_path}")

if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    
    INPUT_CSV = os.path.join(BASE_DIR, "pickleball_dataset", "metadata_dev", "audio_ball_hits.csv")
    OUTPUT_FILE = os.path.join(BASE_DIR, "pickleball_hit_frames.csv")
    
    MY_FPS = 30 
    
    generate_frame_labels(
        input_csv_path=INPUT_CSV,
        output_csv_path=OUTPUT_FILE,
        fps=MY_FPS
    )