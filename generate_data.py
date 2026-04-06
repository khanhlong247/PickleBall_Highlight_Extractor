import os
import shutil
import numpy as np
import librosa
import pandas as pd
import tensorflow as tf
import tensorflow_hub as hub
from scipy.signal import find_peaks
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans  
from tqdm import tqdm
import soundfile as sf

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

FULL_MATCH_PATH = os.path.join(BASE_DIR, "evaluation_data/short_match_9.mp4")
ANCHOR_PATH = os.path.join(BASE_DIR, "samples/cut.wav") 

OUTPUT_BASE = os.path.join(BASE_DIR, "pickleball_test12")
MIC_DIR = os.path.join(OUTPUT_BASE, "mic_dev")
META_DIR = os.path.join(OUTPUT_BASE, "metadata_dev")

SAMPLE_RATE_YAMNET = 16000
SAMPLE_RATE_TRAIN = 32000
WINDOW_SIZE = 0.96
STRIDE = 0.1
SIMILARITY_THRESHOLD = 0.75

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
    try:
        class_map = pd.read_csv(YAMNET_CLASS_MAP_URL)
        target_classes = ["Crowd", "Cheering", "Applause"]
        indices = {}
        for target in target_classes:
            row = class_map[class_map['display_name'] == target]
            if not row.empty:
                indices[target] = row.iloc[0]['index']
        return indices
    except Exception as e:
        print(f"Không lấy được class map: {e}. Sử dụng index mặc định.")
        return {"Crowd": 4, "Cheering": 6, "Applause": 56}

def get_embedding(model, waveform):
    scores, embeddings, spectrogram = model(waveform)
    mean_embedding = None
    mean_scores = None

    if len(embeddings) > 0:
        mean_embedding = np.mean(embeddings.numpy(), axis=0).reshape(1, -1)
    
    if len(scores) > 0:
        mean_scores = np.mean(scores.numpy(), axis=0)
        
    return mean_embedding, mean_scores

def generate_dynamic_template(model, y_full, sr, anchor_path):
    print("\n" + "="*60)
    print("BẮT ĐẦU AUTO-CALIBRATION: ELBOW + ANCHOR-GUIDED")
    print("="*60)
    
    try:
        anchor_wav, _ = librosa.load(anchor_path, sr=sr, mono=True)
        win_samples = int(WINDOW_SIZE * sr)
        if len(anchor_wav) < win_samples:
            anchor_wav = np.pad(anchor_wav, (0, win_samples - len(anchor_wav)))
        else:
            anchor_wav = anchor_wav[:win_samples]
            
        anchor_emb, _ = get_embedding(model, anchor_wav)
        if anchor_emb is None:
            raise ValueError("Không thể trích xuất đặc trưng từ mỏ neo.")
        anchor_vector = anchor_emb[0]
        print(f"[1/5] Load mỏ neo thành công từ: {os.path.basename(anchor_path)}")
    except Exception as e:
        print(f"Lỗi load mỏ neo: {e}")
        return None

    print("[2/5] Dò tìm và trích xuất đặc trưng âm thanh...")
    onset_frames = librosa.onset.onset_detect(y=y_full, sr=sr, backtrack=False)
    onset_times = librosa.frames_to_time(onset_frames, sr=sr)
    
    embeddings = []
    half_window = int((WINDOW_SIZE / 2) * sr)

    for t in tqdm(onset_times, desc="      Đang quét"):
        center_sample = int(t * sr)
        start = max(0, center_sample - half_window)
        end = start + win_samples
        wav_chunk = y_full[start:end]
        
        if len(wav_chunk) < win_samples:
            wav_chunk = np.pad(wav_chunk, (0, win_samples - len(wav_chunk)))

        rms_energy = np.sqrt(np.mean(wav_chunk**2))
        if rms_energy < 0.01:
            continue

        emb, scores = get_embedding(model, wav_chunk)
        if emb is not None and scores is not None:
            if scores[0] > 0.3 or scores[137] > 0.3:
                continue
            embeddings.append(emb[0])

    X = np.array(embeddings)
    if len(X) < 15:
        print("Quá ít tiếng động hợp lệ để phân cụm. Dùng lại mỏ neo gốc.")
        return [np.array(anchor_vector).reshape(1, -1)]

    print(f"[3/5] Chạy K-Means & Tìm K tối ưu (Elbow Method) trên {len(X)} mẫu...")
    max_k = min(15, len(X) // 5)
    if max_k < 3: max_k = 3
    
    inertias = []
    kmeans_models = []
    
    for k in range(2, max_k + 1):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(X)
        inertias.append(kmeans.inertia_)
        kmeans_models.append(kmeans)

    coords = np.vstack((range(2, max_k + 1), inertias)).T
    first_point, last_point = coords[0], coords[-1]
    line_vec = last_point - first_point
    line_vec_norm = line_vec / np.sqrt(np.sum(line_vec**2))
    
    vec_from_first = coords - first_point
    scalar_proj = np.sum(vec_from_first * line_vec_norm, axis=1)
    vec_proj = np.outer(scalar_proj, line_vec_norm)
    vec_to_line = vec_from_first - vec_proj
    dist_to_line = np.sqrt(np.sum(vec_to_line**2, axis=1))
    
    best_k_idx = np.argmax(dist_to_line)
    optimal_k = best_k_idx + 2
    best_kmeans = kmeans_models[best_k_idx]
    labels = best_kmeans.labels_
    
    print(f"Đã tìm thấy K tối ưu = {optimal_k} (Môi trường âm thanh có {optimal_k} loại tạp âm chính).")

    print("[4/5] Dùng Mỏ Neo tìm cụm Tiếng Bóng...")
    best_cluster_idx = -1
    highest_sim = -1
    
    for i in range(optimal_k):
        cluster_vectors = X[labels == i]
        cluster_centroid = np.mean(cluster_vectors, axis=0)
        sim = cosine_similarity([cluster_centroid], [anchor_vector])[0][0]
        count = len(cluster_vectors)
        
        if sim > highest_sim and count >= 5: 
            highest_sim = sim
            best_cluster_idx = i

    if best_cluster_idx == -1:
        print("Cảnh báo: Không có cụm nào phù hợp. Trả về Mỏ neo gốc.")
        return [np.array(anchor_vector).reshape(1, -1)]

    print(f"Đã chốt Cụm {best_cluster_idx} làm mục tiêu (Độ giống với Mỏ neo: {highest_sim:.3f}).")
    print("[5/5] Ép khuôn Dynamic Template bằng Trung Vị (Median)...")
    
    best_cluster_vectors = X[labels == best_cluster_idx]
    master_template = np.median(best_cluster_vectors, axis=0).reshape(1, -1)
    
    print("="*60)
    print("HOÀN TẤT TẠO TEMPLATE ĐỘNG!")
    print("="*60)

    return [master_template]

def scan_match(model, full_audio, sr, templates, threshold=0.7):
    detected_times = []
    crowd_data = []
    class_indices = get_yamnet_class_indices()
    
    win_len = int(WINDOW_SIZE * sr)
    stride_len = int(STRIDE * sr)
    
    if len(full_audio) < win_len:
        print("File âm thanh quá ngắn để quét.")
        return [], [], []

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
            sim_scores = [cosine_similarity(t_emb, emb)[0][0] for t_emb in templates]
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
                    val = scores[cls_idx] if cls_idx < len(scores) else 0.0
                    crowd_entry[cls_name] = val
                crowd_data.append(crowd_entry)
        else:
            scores_over_time.append(0)
            
    return detected_times, np.array(scores_over_time), crowd_data

def process_crowd_noise(crowd_data, duration_sec=10.0):
    if not crowd_data: return pd.DataFrame()
    df = pd.DataFrame(crowd_data)
    df['segment_idx'] = (df['timestamp'] // duration_sec).astype(int)
    grouped = df.groupby('segment_idx').agg({
        'timestamp': 'min', 'Crowd': 'mean', 'Cheering': 'mean', 'Applause': 'mean'
    }).rename(columns={'timestamp': 'segment_start'})
    grouped['segment_end'] = grouped['segment_start'] + duration_sec
    noise_cols = ['Crowd', 'Cheering', 'Applause']
    grouped['crowd_level'] = grouped[noise_cols].max(axis=1).round(4)
    grouped['crowd_type_raw'] = grouped[noise_cols].idxmax(axis=1)
    type_mapping = {'Crowd': 'general', 'Cheering': 'cheering', 'Applause': 'applause'}
    grouped['crowd_type'] = grouped['crowd_type_raw'].map(type_mapping)
    return grouped[['segment_start', 'segment_end', 'crowd_level', 'crowd_type']].reset_index(drop=True)

def refine_peaks(raw_hits):
    print("\n--- Tinh chỉnh vị trí (Peak Picking) ---")
    final_labels = []
    raw_hits.sort(key=lambda x: x['score'], reverse=True)
    kept_hits = []
    for hit in raw_hits:
        is_duplicate = False
        for existing in kept_hits:
            if abs(existing['time'] - hit['time']) < 0.5:
                is_duplicate = True; break
        if not is_duplicate: kept_hits.append(hit)
    
    kept_hits.sort(key=lambda x: x['time'])
    
    for idx, hit in enumerate(kept_hits):
        onset_env = librosa.onset.onset_strength(y=hit['chunk'], sr=SAMPLE_RATE_YAMNET)
        local_peaks = librosa.util.peak_pick(onset_env, pre_max=3, post_max=3, pre_avg=3, post_avg=5, delta=0.5, wait=10)
        if len(local_peaks) > 0:
            best_peak = local_peaks[np.argmax(onset_env[local_peaks])]
            exact_time = hit['time'] + librosa.frames_to_time(best_peak, sr=SAMPLE_RATE_YAMNET)
        else:
            exact_time = hit['time'] + (WINDOW_SIZE / 2)
            
        final_labels.append({
            'hit_id': idx, 'start': round(exact_time - 0.075, 3), 'end': round(exact_time + 0.075, 3),
            'midpoint': round(exact_time, 3), 'similarity': round(hit['score'], 4)
        })
    return final_labels

if __name__ == "__main__":
    import time
    start_time = time.time()
    
    setup_environment()
    
    if not os.path.exists(FULL_MATCH_PATH):
        print(f"Lỗi: Không tìm thấy file trận đấu tại {FULL_MATCH_PATH}")
        exit()

    yamnet_model = load_model()

    print("\n--- Load trận đấu ---")
    y_full, _ = librosa.load(FULL_MATCH_PATH, sr=SAMPLE_RATE_YAMNET, mono=True)

    dynamic_templates = generate_dynamic_template(yamnet_model, y_full, SAMPLE_RATE_YAMNET, ANCHOR_PATH)
    
    if not dynamic_templates:
        print("Lỗi: Không thể tạo Template Động. Video có thể không có tiếng động nào rõ ràng.")
        exit()

    raw_hits, score_arr, crowd_raw_data = scan_match(yamnet_model, y_full, SAMPLE_RATE_YAMNET, dynamic_templates, threshold=SIMILARITY_THRESHOLD)
    print(f"\nTìm thấy {len(raw_hits)} đoạn nghi vấn.")

    print("\n--- Xử lý Crowd Noise ---")
    crowd_df = process_crowd_noise(crowd_raw_data, duration_sec=10.0)
    if not crowd_df.empty:
        crowd_csv_path = os.path.join(META_DIR, "audio_crowd_noise.csv")
        crowd_df.to_csv(crowd_csv_path, index=False)
        print(f"Đã lưu Crowd Noise: {crowd_csv_path}")

    if len(raw_hits) > 0:
        final_labels = refine_peaks(raw_hits)
        print(f"\nKết quả cuối cùng: {len(final_labels)} cú đánh được phát hiện.")
        
        end_time = time.time()
        total_time = end_time - start_time
        print(f"\nTỔNG THỜI GIAN CHẠY: {total_time:.2f} giây (tương đương {total_time/60:.2f} phút)")

        print("\n--- Tạo Dataset Files ---")
        base_name = "generated_match_01"

        df = pd.DataFrame(final_labels)
        if not df.empty:
            csv_path = os.path.join(META_DIR, "audio_ball_hits.csv")
            df = df[['hit_id', 'start', 'end', 'midpoint', 'similarity']]
            df.to_csv(csv_path, index=False)
            print(f"Đã lưu Metadata Hits: {csv_path}")
        
        print(f"\nĐang xử lý và lưu Audio chất lượng cao (32kHz)...")
        wav_out_path = os.path.join(MIC_DIR, f"{base_name}.wav")
        y_high, _ = librosa.load(FULL_MATCH_PATH, sr=SAMPLE_RATE_TRAIN, mono=True)
        sf.write(wav_out_path, y_high, SAMPLE_RATE_TRAIN)
        print(f"Đã lưu Audio: {wav_out_path}")
        print("\nXONG! Dữ liệu đã sẵn sàng trong thư mục 'pickleball_test'.")
    else:
        print("Không tìm thấy cú đánh nào! Hãy thử giảm SIMILARITY_THRESHOLD trong code.")
        end_time = time.time()
        total_time = end_time - start_time
        print(f"\nTỔNG THỜI GIAN CHẠY: {total_time:.2f} giây (tương đương {total_time/60:.2f} phút)")