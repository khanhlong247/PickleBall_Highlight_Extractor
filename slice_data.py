import os
import pandas as pd
import numpy as np
import librosa
import soundfile as sf
from tqdm import tqdm

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

BASE_INPUT = os.path.join(BASE_DIR, "pickleball_dataset")
INPUT_WAV = os.path.join(BASE_INPUT, "mic_dev", "generated_match_01.wav")
INPUT_CSV = os.path.join(BASE_INPUT, "metadata_dev", "generated_match_01.csv")

OUTPUT_DIR = os.path.join(BASE_DIR, "pickleball_sliced") 
MIC_OUT = os.path.join(OUTPUT_DIR, "mic_dev")
META_OUT = os.path.join(OUTPUT_DIR, "metadata_dev")

os.makedirs(MIC_OUT, exist_ok=True)
os.makedirs(META_OUT, exist_ok=True)

SEGMENT_DURATION = 10.0

# 1. Load dữ liệu
print(f"Đang load: {INPUT_WAV}")
y, sr = librosa.load(INPUT_WAV, sr=32000, mono=True) 
df_labels = pd.read_csv(INPUT_CSV)

total_duration = len(y) / sr
num_segments = int(total_duration // SEGMENT_DURATION)
print(f"Audio dài {total_duration:.1f}s. Sẽ cắt thành {num_segments} đoạn nhỏ.")

counts = {"hit": 0, "no_hit": 0}

# 2. Bắt đầu cắt
for i in tqdm(range(num_segments)):
    start_t = i * SEGMENT_DURATION
    end_t = start_t + SEGMENT_DURATION
    
    filename_base = f"segment_{i:04d}"
    wav_path = os.path.join(MIC_OUT, f"{filename_base}.wav")
    csv_path = os.path.join(META_OUT, f"{filename_base}.csv")
    
    # Lấy các nhãn nằm trong khoảng thời gian này
    segment_labels = df_labels[
        (df_labels['start'] < end_t) & 
        (df_labels['end'] > start_t)
    ].copy()
    
    # Cắt Audio
    start_sample = int(start_t * sr)
    end_sample = int(end_t * sr)
    y_segment = y[start_sample:end_sample]
    
    if len(y_segment) < sr * 1.0: continue 
        
    sf.write(wav_path, y_segment, sr)
    
    csv_content = "class,start,end,ele,azi\n"
    
    if not segment_labels.empty:
        counts["hit"] += 1
        for _, row in segment_labels.iterrows():
            new_start = max(0.0, row['start'] - start_t)
            new_end = min(SEGMENT_DURATION, row['end'] - start_t)
            
            if new_end > new_start:
                csv_content += f"hit,{new_start:.3f},{new_end:.3f},0,0\n"
    else:
        counts["no_hit"] += 1
    
    with open(csv_path, "w") as f:
        f.write(csv_content)

print("\nHOÀN TẤT!")
print(f"Dữ liệu mới nằm tại: {OUTPUT_DIR}")
print(f"Chi tiết: {counts['hit']} file có bóng, {counts['no_hit']} file nhiễu nền.")