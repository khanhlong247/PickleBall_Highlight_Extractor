import os
import shutil
import numpy as np
import librosa
import pandas as pd
import tensorflow as tf
import tensorflow_hub as hub
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans  
from tqdm import tqdm
import soundfile as sf

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FULL_MATCH_PATH = os.path.join(BASE_DIR, "evaluation_data/short_match_7.mp4")
ANCHOR_PATH = os.path.join(BASE_DIR, "samples/cut.wav") 
SAMPLE_RATE_YAMNET = 16000
WINDOW_SIZE = 0.96

def load_model():
    return hub.load('https://tfhub.dev/google/yamnet/1')

def get_embedding(model, waveform):
    scores, embeddings, _ = model(waveform)
    if len(embeddings) > 0:
        return np.mean(embeddings.numpy(), axis=0).reshape(1, -1), np.mean(scores.numpy(), axis=0)
    return None, None

def generate_dynamic_template(model, y_full, sr, anchor_path, num_templates=4):
    print("\n" + "="*60)
    print("AUTO-CALIBRATION (ELBOW + ANCHOR) + QUALITY SCORE")
    print("="*60)
    
    anchor_wav, _ = librosa.load(anchor_path, sr=sr, mono=True)
    win_samples = int(WINDOW_SIZE * sr)
    anchor_wav = librosa.util.fix_length(anchor_wav, size=win_samples)
    anchor_vector, _ = get_embedding(model, anchor_wav)

    print("[2/5] Trích xuất đặc trưng & tính toán độ sắc nét (Onset)...")
    onset_frames = librosa.onset.onset_detect(y=y_full, sr=sr, backtrack=False)
    onset_times = librosa.frames_to_time(onset_frames, sr=sr)
    
    candidates = []
    half_window = int(win_samples / 2)

    for t in tqdm(onset_times, desc="      Phân tích candidates"):
        center = int(t * sr)
        start, end = max(0, center - half_window), center + half_window
        chunk = y_full[start:end]
        chunk = librosa.util.fix_length(chunk, size=win_samples)

        o_env = librosa.onset.onset_strength(y=chunk, sr=sr)
        max_onset = np.max(o_env)

        if np.sqrt(np.mean(chunk**2)) < 0.01: continue

        emb, scores = get_embedding(model, chunk)
        if emb is not None and scores is not None:
            if scores[0] > 0.3 or scores[137] > 0.3: continue
            
            candidates.append({
                'embedding': emb[0],
                'onset_strength': max_onset
            })

    if len(candidates) < 15:
        print("Quá ít candidate. Trả về mỏ neo.")
        return [anchor_vector], 0.0

    X = np.vstack([c['embedding'] for c in candidates])

    max_k = min(15, len(X) // 5)
    inertias = []
    models = []
    for k in range(2, max_k + 1):
        km = KMeans(n_clusters=k, random_state=42, n_init=10).fit(X)
        inertias.append(km.inertia_)
        models.append(km)

    coords = np.vstack((range(2, max_k + 1), inertias)).T
    line_vec = coords[-1] - coords[0]
    line_vec_norm = line_vec / np.sqrt(np.sum(line_vec**2))
    vec_from_first = coords - coords[0]
    dist_to_line = np.sqrt(np.sum((vec_from_first - np.outer(np.sum(vec_from_first * line_vec_norm, axis=1), line_vec_norm))**2, axis=1))
    
    optimal_k = np.argmax(dist_to_line) + 2
    best_kmeans = models[np.argmax(dist_to_line)]
    labels = best_kmeans.labels_
    print(f"K tối ưu: {optimal_k}")

    best_main_cluster = -1
    max_sim = -1
    for i in range(optimal_k):
        centroid = np.mean(X[labels == i], axis=0)
        sim = cosine_similarity([centroid], anchor_vector)[0][0]
        if sim > max_sim:
            max_sim = sim
            best_main_cluster = i

    ball_candidates = [c for i, c in enumerate(candidates) if labels[i] == best_main_cluster]
    X_ball = np.vstack([c['embedding'] for c in ball_candidates])

    sub_kmeans = KMeans(n_clusters=num_templates, random_state=42, n_init=10).fit(X_ball)
    
    final_templates = []
    representative_qualities = []

    for cluster_id in range(num_templates):
        sub_mask = sub_kmeans.labels_ == cluster_id
        sub_candidates = [c for j, c in enumerate(ball_candidates) if sub_mask[j]]
        
        if not sub_candidates: continue

        best_rep = max(sub_candidates, key=lambda x: x['onset_strength'])
        final_templates.append(best_rep['embedding'].reshape(1, -1))
        representative_qualities.append(best_rep['onset_strength'])

    quality_score = np.mean(representative_qualities)
    
    print(f"Đã chọn {len(final_templates)} đại diện. Quality Score: {quality_score:.2f}")
    return final_templates, quality_score

if __name__ == "__main__":
    model = load_model()
    
    import time
    start_time = time.time()
    
    y, _ = librosa.load(FULL_MATCH_PATH, sr=SAMPLE_RATE_YAMNET)
    
    templates, q_score = generate_dynamic_template(model, y, SAMPLE_RATE_YAMNET, ANCHOR_PATH)
    
    end_time = time.time()
    total_time = end_time - start_time
    print(f"\nTỔNG THỜI GIAN CHẠY: {total_time:.2f} giây (tương đương {total_time/60:.2f} phút)")
    
    print(f"\nKẾT QUẢ CUỐI CÙNG:")
    print(f"- Số lượng templates: {len(templates)}")
    print(f"- QUALITY SCORE (Onset Avg): {q_score:.2f}")