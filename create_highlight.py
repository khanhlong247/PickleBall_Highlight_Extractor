import os
import pandas as pd
from moviepy import VideoFileClip, concatenate_videoclips
from tqdm import tqdm

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

VIDEO_PATH = os.path.join(BASE_DIR, "input_sample_match.mp4")
MASTER_CSV = os.path.join(BASE_DIR, "pickleball_test1", "metadata_dev", "audio_ball_hits.csv")
OUTPUT_VIDEO = os.path.join(BASE_DIR, "result_video", "highlight.mp4")

# --- CẤU HÌNH THUẬT TOÁN NHỊP ĐỘ ---
MIN_RALLY_HITS = 5        # Một highlight cần ít nhất bao nhiêu lần chạm vợt liên tiếp
MIN_INTERVAL = 0.2        # Thời gian tối thiểu giữa 2 lần chạm (chống nhiễu nhồi bóng)
MAX_INTERVAL = 2.0        # Thời gian tối đa giữa 2 lần chạm (bóng chết)
PADDING_START = 2.0       # Lùi lại 2 giây trước tiếng đập đầu tiên để thấy cầu thủ chuẩn bị
PADDING_END = 2.0         # Cộng thêm 2 giây sau tiếng đập cuối cùng để thấy kết quả pha bóng

def find_dynamic_highlights(df):
    timestamps = sorted(df['midpoint'].tolist())
    
    highlights = []
    current_chain = []

    for t in timestamps:
        if not current_chain:
            current_chain.append(t)
            continue
            
        time_since_last = t - current_chain[-1]
        
        if MIN_INTERVAL <= time_since_last <= MAX_INTERVAL:
            current_chain.append(t)
        
        elif time_since_last < MIN_INTERVAL:
            continue
            
        else:
            if len(current_chain) >= MIN_RALLY_HITS:
                highlights.append({
                    'start_hit': current_chain[0],
                    'end_hit': current_chain[-1],
                    'num_hits': len(current_chain)
                })
            current_chain = [t]

    if len(current_chain) >= MIN_RALLY_HITS:
        highlights.append({
            'start_hit': current_chain[0],
            'end_hit': current_chain[-1],
            'num_hits': len(current_chain)
        })
        
    return highlights

def create_highlight_video():
    if not os.path.exists(VIDEO_PATH):
        print(f"Video not found: {VIDEO_PATH}")
        return
    if not os.path.exists(MASTER_CSV):
        print(f"CSV gốc không tồn tại: {MASTER_CSV}")
        return

    print("Đang phân tích nhịp độ trận đấu (Pace Analysis)...")
    df = pd.read_csv(MASTER_CSV)
    
    rallies = find_dynamic_highlights(df)
    
    if not rallies:
        print("Không tìm thấy pha đôi công nào đủ điều kiện highlight.")
        return

    print(f"\nTuyệt vời! Tìm thấy {len(rallies)} pha bóng bền (Rallies):")
    for i, r in enumerate(rallies):
        print(f"  - Pha {i+1}: {r['num_hits']} chạm | Từ {r['start_hit']:.1f}s đến {r['end_hit']:.1f}s")

    try:
        source_clip = VideoFileClip(VIDEO_PATH)
        total_duration = source_clip.duration
        highlight_clips = []
        
        print("\nĐang cắt các đoạn highlight (có thêm Padding)...")
        for r in tqdm(rallies, desc="Cắt clip"):
            start_cut = max(0, r['start_hit'] - PADDING_START)
            end_cut = min(total_duration, r['end_hit'] + PADDING_END)
            
            clip = source_clip.subclipped(start_cut, end_cut)
            highlight_clips.append(clip)

        print(f"\nĐang ghép nối {len(highlight_clips)} pha bóng...")
        final_clip = concatenate_videoclips(highlight_clips)

        os.makedirs(os.path.dirname(OUTPUT_VIDEO), exist_ok=True)
        print(f"Đang render video ra: {OUTPUT_VIDEO}")
        
        final_clip.write_videofile(
            OUTPUT_VIDEO, 
            codec='libx264', 
            audio_codec='aac',
            audio=True,
            temp_audiofile='temp-audio.m4a',
            remove_temp=True,
            fps=source_clip.fps
        )
        
        source_clip.close()
        final_clip.close()
        
        print("\nXONG! Video highlight chất lượng cao đã sẵn sàng.")
        
    except Exception as e:
        print(f"Lỗi trong quá trình cắt video: {e}")

if __name__ == "__main__":
    create_highlight_video()