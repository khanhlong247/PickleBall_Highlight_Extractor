import os
import pandas as pd
from tqdm import tqdm

META_DIR = r"D:\Code\Visual studio code\pickleball_audio_extract\pickleball_dataset_sliced\metadata_dev"

def count_hits_in_csvs(directory):
    if not os.path.exists(directory):
        print(f"Lỗi: Thư mục không tồn tại: {directory}")
        return

    csv_files = [f for f in os.listdir(directory) if f.endswith('.csv')]
    
    if not csv_files:
        print("Không tìm thấy file CSV nào trong thư mục.")
        return

    print(f"Đang phân tích {len(csv_files)} file CSV...")
    
    stats = []
    total_hits_all_files = 0
    empty_files_count = 0

    for filename in tqdm(csv_files, desc="Đang đếm"):
        filepath = os.path.join(directory, filename)
        
        try:
            df = pd.read_csv(filepath)
            
            if 'class' in df.columns:
                num_hits = len(df[df['class'] == 'hit'])
            else:
                num_hits = len(df)

            stats.append((filename, num_hits))
            total_hits_all_files += num_hits
            
            if num_hits == 0:
                empty_files_count += 1
                
        except pd.errors.EmptyDataError:
            stats.append((filename, 0))
            empty_files_count += 1
        except Exception as e:
            print(f"Lỗi đọc file {filename}: {e}")

    print("\n" + "="*40)
    print("BẢNG THỐNG KÊ CHI TIẾT")
    print("="*40)
    print(f"{'Tên File':<20} | {'Số lượng Hit'}")
    print("-" * 35)
    
    stats.sort(key=lambda x: x[1], reverse=True) 
    
    for name, count in stats:
        if count > 0:
            print(f"{name:<20} | {count}")
    
    print("-" * 35)
    print(f"TỔNG HỢP:")
    print(f"   - Tổng số file CSV: {len(csv_files)}")
    print(f"   - Số file CÓ bóng ('hit' > 0): {len(csv_files) - empty_files_count}")
    print(f"   - Số file KHÔNG bóng (nhiễu nền): {empty_files_count}")
    print(f"   - Tổng cộng tất cả các cú đánh (Total Hits): {total_hits_all_files}")
    print("="*40)

if __name__ == "__main__":
    count_hits_in_csvs(META_DIR)