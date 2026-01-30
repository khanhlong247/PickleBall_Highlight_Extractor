import os
import shutil
import numpy as np
import librosa
import pandas as pd
import tensorflow as tf
import tensorflow_hub as hub
from scipy.signal import find_peaks
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
import soundfile as sf

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

FULL_MATCH_PATH = os.path.join(BASE_DIR, "raw_audio/full_match.wav")
TEMPLATE_PATHS = [
    os.path.join(BASE_DIR, "samples/cut.wav"),
    os.path.join(BASE_DIR, "samples/cut1.wav")
]

OUTPUT_BASE = os.path.join(BASE_DIR, "pickleball_dataset")
MIC_DIR = os.path.join(OUTPUT_BASE, "mic_dev")
META_DIR = os.path.join(OUTPUT_BASE, "metadata_dev")

SAMPLE_RATE_YAMNET = 16000
SAMPLE_RATE_TRAIN = 32000
WINDOW_SIZE = 0.96
STRIDE = 0.1
SIMILARITY_THRESHOLD = 0.75

# URL file map class của YAMNet để tìm index của Crowd noise
YAMNET_CLASS_MAP_URL = "https://raw.githubusercontent.com/tensorflow/models/master/research/audioset/yamnet/yamnet_class_map.csv"

def setup_environment():
    os.makedirs(MIC_DIR, exist_ok=True)
    os.makedirs(META_DIR, exist_ok=True)
    print(f"Đã setup môi trường. Output folder: {OUTPUT_BASE}")

def load_model():
    print("Đang load YAMNet...")
    model = hub.load('https://tfhub.dev/google/yamnet/1')
    print("Đã load YAMNet.")
    return model

def get_yamnet_class_indices():
    """Lấy index của các lớp âm thanh đám đông từ file map của YAMNet"""
    try:
        class_map = pd.read_csv(YAMNET_CLASS_MAP_URL)
        target_classes = ["Crowd", "Cheering", "Applause"]
        indices = {}
        for target in target_classes:
            row = class_map[class_map['display_name'] == target]
            if not row.empty:
                indices[target] = row.iloc[0]['index']
        print(f"ℹ️ Class Indices: {indices}")
        return indices
    except Exception as e:
        print(f"Không lấy được class map: {e}. Sử dụng index mặc định (có thể không chính xác).")
        return {"Crowd": 4, "Cheering": 6, "Applause": 56}

def get_embedding(model, waveform):
    """
    Lấy vector đặc trưng (embeddings) và điểm số phân loại (scores).
    """
    scores, embeddings, spectrogram = model(waveform)
    
    mean_embedding = None
    mean_scores = None

    if len(embeddings) > 0:
        mean_embedding = np.mean(embeddings.numpy(), axis=0).reshape(1, -1)
    
    if len(scores) > 0:
        mean_scores = np.mean(scores.numpy(), axis=0)
        
    return mean_embedding, mean_scores

def extract_templates(model, template_paths):
    templates = []
    print("--- Xử lý file mẫu ---")
    
    for path in template_paths:
        if not os.path.exists(path):
            print(f"Cảnh báo: Không tìm thấy file mẫu: {path}")
            continue

        try:
            wav, _ = librosa.load(path, sr=SAMPLE_RATE_YAMNET, mono=True)
            
            if len(wav) > int(WINDOW_SIZE * SAMPLE_RATE_YAMNET):
                center = np.argmax(np.abs(wav))
                start = max(0, center - int(0.48 * SAMPLE_RATE_YAMNET))
                end = start + int(WINDOW_SIZE * SAMPLE_RATE_YAMNET)
                wav = wav[start:end]
            
            if len(wav) < int(WINDOW_SIZE * SAMPLE_RATE_YAMNET):
                wav = np.pad(wav, (0, int(WINDOW_SIZE * SAMPLE_RATE_YAMNET) - len(wav)))
            
            emb, _ = get_embedding(model, wav)
            
            if emb is not None:
                templates.append(emb)
                print(f"  + Đã trích xuất đặc trưng mẫu: {os.path.basename(path)}")
        except Exception as e:
            print(f"  x Lỗi file {path}: {e}")
            
    return templates

def scan_match(model, full_audio, sr, templates, threshold=0.7):
    detected_times = []
    crowd_data = []
    
    class_indices = get_yamnet_class_indices()
    
    win_len = int(WINDOW_SIZE * sr)
    stride_len = int(STRIDE * sr)
    
    if len(full_audio) < win_len:
        print("File âm thanh quá ngắn để quét.")
        return [], []

    num_steps = (len(full_audio) - win_len) // stride_len
    
    print(f"\nBắt đầu quét {len(full_audio)/sr:.1f} giây ({num_steps} bước)...")
    
    scores_over_time = []
    
    for i in tqdm(range(num_steps), desc="Scanning"):
        start_sample = i * stride_len
        end_sample = start_sample + win_len
        chunk = full_audio[start_sample:end_sample]
        current_time = i * STRIDE
        
        emb, scores = get_embedding(model, chunk)
        
        if emb is not None:
            sim_scores = []
            for t_emb in templates:
                sim = cosine_similarity(t_emb, emb)[0][0]
                sim_scores.append(sim)
            
            avg_sim = np.mean(sim_scores) if sim_scores else 0
            
            scores_over_time.append(avg_sim)
            
            if avg_sim >= threshold:
                detected_times.append({
                    'time': current_time,
                    'score': avg_sim,
                    'chunk': chunk
                })
            
            if scores is not None:
                crowd_entry = {'timestamp': current_time}
                for cls_name, cls_idx in class_indices.items():
                    if cls_idx < len(scores):
                        crowd_entry[cls_name] = scores[cls_idx]
                    else:
                        crowd_entry[cls_name] = 0.0
                crowd_data.append(crowd_entry)
                
        else:
            scores_over_time.append(0)
            
    return detected_times, np.array(scores_over_time), crowd_data

def process_crowd_noise(crowd_data, duration_sec=10.0):
    """Tính toán level và type cho Crowd Noise"""
    if not crowd_data:
        return pd.DataFrame()
    
    df = pd.DataFrame(crowd_data)
    
    df['segment_idx'] = (df['timestamp'] // duration_sec).astype(int)
    
    grouped = df.groupby('segment_idx').agg({
        'timestamp': 'min',
        'Crowd': 'mean',
        'Cheering': 'mean',
        'Applause': 'mean'
    }).rename(columns={'timestamp': 'segment_start'})
    
    grouped['segment_end'] = grouped['segment_start'] + duration_sec
    
    noise_cols = ['Crowd', 'Cheering', 'Applause']
    
    grouped['crowd_level'] = grouped[noise_cols].max(axis=1).round(4)
    
    grouped['crowd_type_raw'] = grouped[noise_cols].idxmax(axis=1)
    
    type_mapping = {
        'Crowd': 'general',
        'Cheering': 'cheering',
        'Applause': 'applause'
    }
    grouped['crowd_type'] = grouped['crowd_type_raw'].map(type_mapping)
    
    output_df = grouped[['segment_start', 'segment_end', 'crowd_level', 'crowd_type']].reset_index(drop=True)
    
    return output_df

def refine_peaks(raw_hits):
    print("\n--- Tinh chỉnh vị trí (Peak Picking) ---")
    final_labels = []
    
    raw_hits.sort(key=lambda x: x['score'], reverse=True)
    
    kept_hits = []
    for hit in raw_hits:
        coarse_time = hit['time']
        is_duplicate = False
        for existing in kept_hits:
            if abs(existing['time'] - coarse_time) < 0.5:
                is_duplicate = True
                break
        if not is_duplicate:
            kept_hits.append(hit)
    
    kept_hits.sort(key=lambda x: x['time'])
    
    for idx, hit in enumerate(kept_hits):
        coarse_time = hit['time']
        chunk = hit['chunk']
        onset_env = librosa.onset.onset_strength(y=chunk, sr=SAMPLE_RATE_YAMNET)
        
        local_peaks = librosa.util.peak_pick(onset_env, pre_max=3, post_max=3, pre_avg=3, post_avg=5, delta=0.5, wait=10)
        
        if len(local_peaks) > 0:
            best_peak_idx = local_peaks[np.argmax(onset_env[local_peaks])]
            offset_time = librosa.frames_to_time(best_peak_idx, sr=SAMPLE_RATE_YAMNET)
            exact_time = coarse_time + offset_time
        else:
            exact_time = coarse_time + (WINDOW_SIZE / 2)
            
        label_entry = {
            'hit_id': idx, 
            'start': round(exact_time - 0.075, 3),
            'end': round(exact_time + 0.075, 3),
            'midpoint': round(exact_time, 3),
            'similarity': round(hit['score'], 4)
        }
        final_labels.append(label_entry)
    
    return final_labels

if __name__ == "__main__":
    setup_environment()
    
    if not os.path.exists(FULL_MATCH_PATH):
        print(f"Lỗi: Không tìm thấy file trận đấu tại {FULL_MATCH_PATH}")
        exit()

    yamnet_model = load_model()

    template_embeddings = extract_templates(yamnet_model, TEMPLATE_PATHS)
    if not template_embeddings:
        print("Lỗi: Không tạo được mẫu nào. Kiểm tra lại file cut.wav/cut1.wav")
        exit()

    print("\n--- Load trận đấu (để quét) ---")
    y_full, _ = librosa.load(FULL_MATCH_PATH, sr=SAMPLE_RATE_YAMNET, mono=True)

    # 1. Quét tìm Hit & Thu thập Crowd Data
    raw_hits, score_arr, crowd_raw_data = scan_match(yamnet_model, y_full, SAMPLE_RATE_YAMNET, template_embeddings, threshold=SIMILARITY_THRESHOLD)
    print(f"\nTìm thấy {len(raw_hits)} đoạn nghi vấn.")

    # 2. Xử lý & Lưu Crowd Noise
    print("\n--- Xử lý Crowd Noise ---")
    crowd_df = process_crowd_noise(crowd_raw_data, duration_sec=10.0)
    if not crowd_df.empty:
        crowd_csv_path = os.path.join(META_DIR, "audio_crowd_noise.csv")
        crowd_df.to_csv(crowd_csv_path, index=False)
        print(f"Đã lưu Crowd Noise: {crowd_csv_path}")
        print(crowd_df.head(3))

    # 3. Xử lý & Lưu Ball Hits
    if len(raw_hits) > 0:
        final_labels = refine_peaks(raw_hits)
        print(f"\nKết quả cuối cùng: {len(final_labels)} cú đánh được phát hiện.")

        print("\n--- Tạo Dataset Files ---")
        base_name = "generated_match_01"

        df = pd.DataFrame(final_labels)
        if not df.empty:
            csv_path = os.path.join(META_DIR, "audio_ball_hits.csv")
            df = df[['hit_id', 'start', 'end', 'midpoint', 'similarity']]
            df.to_csv(csv_path, index=False)
            print(f"Đã lưu Metadata Hits: {csv_path}")
            print(df.head())
        
        print(f"\nĐang xử lý và lưu Audio chất lượng cao (32kHz)...")
        wav_out_path = os.path.join(MIC_DIR, f"{base_name}.wav")
        
        y_high, _ = librosa.load(FULL_MATCH_PATH, sr=SAMPLE_RATE_TRAIN, mono=True)
        sf.write(wav_out_path, y_high, SAMPLE_RATE_TRAIN)
        print(f"Đã lưu Audio: {wav_out_path}")
        
        print("\nXONG! Dữ liệu đã sẵn sàng trong thư mục 'pickleball_dataset'.")
        
    else:
        print("Không tìm thấy cú đánh nào! Hãy thử giảm SIMILARITY_THRESHOLD trong code.")