# Trích xuất tiếng đập bóng và tạo video Highlight

## Luồng hoạt động:

- Bước 1: Tạo file âm thanh Raw từ video.

- Bước 2: Trich xuất tiếng đập bóng từ âm thanh Raw và tạo metadata.

- Bước 3: Chia nhỏ file âm thanh và mapping các đoạn âm thanh đập bóng.

- Bước 4: Tổng hợp chi tiết số lượng các pha đập bóng.

- Bước 5: Tạo video Highlight.

## Chi tiết hoạt động:

### Tạo môi trường ảo

```
# Tạo môi trường tên là 'audio_env' với Python 3.10
conda create -n audio_env python=3.10 -y

# Kích hoạt môi trường
conda activate audio_env
```

### Cài đặt các thư viện hệ thống cần thiết

```
# Cài FFmpeg cho xử lý video/audio
conda install -c conda-forge ffmpeg -y

# Cài CUDA Toolkit và cuDNN (Hỗ trợ GPU cho TensorFlow)
conda install -c conda-forge cudatoolkit=11.8.0 cudnn=8.9.2 -y
```

### Cài đặt các thư viện Python

```
pip install -r requirements.txt
```

### Bước 1: Tạo file âm thanh Raw từ video

Chỉnh sửa đường dẫn file video đầu vào và nơi lưu file âm thanh đầu ra

```
python convert_audio.py
```

### Bước 2: Trich xuất tiếng đập bóng từ âm thanh Raw và tạo metadata

```
python generate_data.py
```

Hoặc

Chạy file `extract_yamnet.ipynb` trong Kaggle. Tải 2 file `cut.wav` và `cut1.wav` cùng với file .wav được tạo từ Bước 1 lên Kaggle

Chạy từng cell, và tải file .zip kết quả về folder dự án và giải nén.

### Bước 3: Chia nhỏ file âm thanh và mapping các đoạn âm thanh đập bóng

Chỉnh sửa đường dẫn và chạy code:

```
python slice_data.py
```

### Bước 4: Tổng hợp chi tiết số lượng các pha đập bóng

Xem chi tiết số lượng các pha đập bóng trong từng file âm thanh được cắt ra từ bước 3:

```
python count_hits.py
```

### Bước 5: Tạo video Highlight

Dựa trên số lượng các pha đập bóng vừa tổng hợp từ bước 4, chọn ra số lượng pha đập bóng tối thiểu của 1 pha Highlight. Sau đó sửa biến `MIN_HITS` nếu cần.

Chạy code sau để tự động cắt ghép video Highlight:

```
python create_highlight.py
```