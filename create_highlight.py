import os
import pandas as pd
from moviepy import VideoFileClip, concatenate_videoclips
from tqdm import tqdm

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

VIDEO_PATH = os.path.join(BASE_DIR, "raw_video/videoplayback.mp4")
BASE_INPUT = os.path.join(BASE_DIR, "pickleball_sliced")
META_DIR = os.path.join(BASE_INPUT, "metadata_dev")
OUTPUT_VIDEO = os.path.join(BASE_DIR, "result_audio/hightlight.mp4")

SEGMENT_DURATION = 10.0
MIN_HITS = 5

def create_highlight_video():
    if not os.path.exists(VIDEO_PATH):
        print(f"Video not found: {VIDEO_PATH}")
        return
    if not os.path.exists(META_DIR):
        print(f"CSV directory not found: {META_DIR}")
        return

    # Find valid segments (>= 5 hits)
    print("Scanning CSV files for high activity segments...")
    csv_files = [f for f in os.listdir(META_DIR) if f.endswith('.csv')]
    
    high_activity_indices = []

    for filename in tqdm(csv_files, desc="Analyzing CSV"):
        try:
            filepath = os.path.join(META_DIR, filename)
            df = pd.read_csv(filepath)
            
            # Count hits
            if 'class' in df.columns:
                num_hits = len(df[df['class'] == 'hit'])
            else:
                num_hits = len(df)
            
            if num_hits >= MIN_HITS:
                idx_str = filename.split('_')[-1].split('.')[0]
                idx = int(idx_str)
                high_activity_indices.append((idx, num_hits))
                
        except Exception as e:
            print(f"Error reading file {filename}: {e}")

    high_activity_indices.sort(key=lambda x: x[0])

    if not high_activity_indices:
        print(f"No segments found with >= {MIN_HITS} hits.")
        return

    print(f"\nFound {len(high_activity_indices)} exciting segments.")
    
    # Process Video (Cut and Concatenate)
    try:
        # Load source video
        source_clip = VideoFileClip(VIDEO_PATH)
        total_duration = source_clip.duration
        
        highlight_clips = []
        
        print("\nCutting video...")
        for idx, hits in tqdm(high_activity_indices, desc="Processing Clips"):
            # Mapping Timestamp
            start_time = idx * SEGMENT_DURATION
            end_time = start_time + SEGMENT_DURATION
            
            # Check boundaries
            if start_time >= total_duration:
                continue
            if end_time > total_duration:
                end_time = total_duration
            
            clip = source_clip.subclipped(start_time, end_time)
            
            highlight_clips.append(clip)

        # Concatenate
        print(f"\nMerging {len(highlight_clips)} clips...")
        final_clip = concatenate_videoclips(highlight_clips, method="compose")

        # Export
        print(f"Rendering video: {OUTPUT_VIDEO}")
        print("Please wait, rendering may take a while...")
        
        final_clip.write_videofile(
            OUTPUT_VIDEO, 
            codec='libx264', 
            audio_codec='aac',
            temp_audiofile='temp-audio.m4a',
            remove_temp=True,
            fps=source_clip.fps
        )
        
        source_clip.close()
        final_clip.close()
        
        print("\nDONE! Highlight video is ready.")
        
    except Exception as e:
        print(f"Error processing video: {e}")

if __name__ == "__main__":
    create_highlight_video()