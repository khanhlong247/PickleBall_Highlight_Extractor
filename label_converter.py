import os
import pandas as pd
import math
from tqdm import tqdm

def generate_frame_labels(metadata_dir, output_csv_path, fps=30, segment_duration=10.0):
    
    if not os.path.exists(metadata_dir):
        print(f"Lỗi: Không tìm thấy thư mục {metadata_dir}")
        return

    csv_files = sorted([f for f in os.listdir(metadata_dir) if f.startswith("segment_") and f.endswith(".csv")])
    
    if not csv_files:
        print("Không tìm thấy file CSV nào.")
        return

    print(f"Đang xử lý {len(csv_files)} file với FPS={fps}...")
    
    hit_frames_global = []

    for filename in tqdm(csv_files, desc="Converting"):
        try:
            seg_idx_str = filename.split('_')[1].split('.')[0]
            seg_idx = int(seg_idx_str)
            
            global_frame_offset = int(seg_idx * segment_duration * fps)
            
            filepath = os.path.join(metadata_dir, filename)
            df = pd.read_csv(filepath)
            
            if df.empty or 'class' not in df.columns:
                continue

            hits = df[df['class'] == 'hit']
            
            for _, row in hits.iterrows():
                t_start = row['start']
                t_end = row['end']
                
                f_start_local = int(math.floor(t_start * fps))
                f_end_local = int(math.floor(t_end * fps))
                
                for f_local in range(f_start_local, f_end_local + 1):
                    f_global = global_frame_offset + f_local
                    hit_frames_global.append(f_global)
                    
        except Exception as e:
            print(f"⚠️ Lỗi xử lý file {filename}: {e}")

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
    INPUT_META_DIR = os.path.join(BASE_DIR, "pickleball_sliced", "metadata_dev")
    OUTPUT_FILE = os.path.join(BASE_DIR, "pickleball_hit_frames.csv")
    
    MY_FPS = 10
    
    generate_frame_labels(
        metadata_dir=INPUT_META_DIR,
        output_csv_path=OUTPUT_FILE,
        fps=MY_FPS
    )