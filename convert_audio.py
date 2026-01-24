import subprocess
import shutil
import os
import sys

def mp4_to_wav(input_mp4, output_wav, sample_rate=16000, mono=True):
    
    ffmpeg_path = shutil.which("ffmpeg")
    if ffmpeg_path is None:
        print("Lỗi: Không tìm thấy 'ffmpeg' trong hệ thống.")
        print("Hãy cài đặt FFmpeg và thêm vào biến môi trường PATH.")
        sys.exit(1)

    if not os.path.exists(input_mp4):
        print(f"Lỗi: Không tìm thấy file đầu vào tại: {input_mp4}")
        sys.exit(1)

    output_dir = os.path.dirname(output_wav)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print(f"Đang xử lý: {input_mp4} -> {output_wav}")

    cmd = [
        ffmpeg_path,
        "-y",
        "-stats",
        "-loglevel", "info",
        "-i", input_mp4,
        "-vn",
        "-ac", "1" if mono else "2",
        "-ar", str(sample_rate),
        "-f", "wav",
        output_wav
    ]

    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding='utf-8',
            errors='replace'
        )

        for line in process.stdout:
            print(line, end="")

        process.wait()

        if process.returncode != 0:
            raise RuntimeError("FFmpeg trả về mã lỗi (khác 0).")

        print(f"\nConvert thành công! File lưu tại: {output_wav}")

    except Exception as e:
        print(f"\nCó lỗi xảy ra: {e}")

if __name__ == "__main__":
    
    INPUT_FILE = r"D:\Code\Visual studio code\pickleball_audio_extract\pickleball_sample.mp4"
    OUTPUT_FILE = r"D:\Code\Visual studio code\pickleball_audio_extract\pickleball_sample.wav"

    mp4_to_wav(
        input_mp4=INPUT_FILE,
        output_wav=OUTPUT_FILE,
        sample_rate=16000,
        mono=True
    )