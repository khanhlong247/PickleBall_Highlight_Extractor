# Pickleball Highlight Extractor

This repository contains a suite of tools to automatically detect pickleball hits from a video's audio track and generate a highlight reel of the best rallies. Using the YAMNet deep learning model for audio event detection, the system can intelligently identify the sound of a ball being hit and distinguish it from background noise.

## Features

-   **Automatic Hit Detection:** Leverages Google's YAMNet model to scan audio and identify timestamps of pickleball hits.
-   **Dynamic Audio Calibration:** An advanced auto-calibration mechanism uses K-Means clustering and an "anchor" sound sample to create a dynamic template for hit sounds. This adapts the detection to the specific audio environment of each match, improving accuracy.
-   **Rally Identification:** Analyzes the pace and timing between hits to identify exciting rallies, filtering out isolated hits or practice serves.
-   **Highlight Video Generation:** Automatically clips the most intense rallies from the source video and stitches them together into a single, seamless highlight file.
-   **Audio Quality Scoring:** Includes a script to analyze the input audio and generate a "Quality Score," indicating how clear and distinct the ball hits are.
-   **Data Processing Utilities:** Provides scripts for slicing long audio files into segments, converting video to audio, and managing labels.

## Workflow

The process of creating a highlight video involves two main steps:

1.  **Detecting Hits:** Run `generate_data.py` on a full match video. This script analyzes the audio, identifies all ball hit timestamps, and saves them to a CSV file (`pickleball_test1/metadata_dev/audio_ball_hits.csv`).
2.  **Creating the Highlight:** Run `create_highlight.py`. This script reads the timestamp CSV and the original video file to find the best rallies, clips them, and concatenates them into a final highlight video.

## Key Scripts Explained

### `generate_data.py`
This is the core detection engine. It takes a full video file as input and performs the following actions:
1.  **Loads Audio:** Extracts the audio from the input video file.
2.  **Auto-Calibration:** Scans the audio for all prominent sound events. It then uses K-Means clustering to group these sounds. By comparing these clusters to a provided "anchor" sample of a clean ball hit, it identifies the primary cluster of ball hits for the specific match. A dynamic template is then generated from the median of this cluster.
3.  **Scanning:** Slides a window across the entire audio track, comparing each segment to the dynamic template to find matches.
4.  **Refinement:** Refines the detected timestamps by finding the precise peak of the sound's onset, ensuring perfect timing.
5.  **Output:** Generates a CSV file containing the `start`, `end`, and `midpoint` timestamps for every detected hit.

### `create_highlight.py`
This script takes the generated timestamp CSV and the original video to produce the final highlight reel.
-   **Pace Analysis:** It groups consecutive hits into "rallies" based on configurable parameters:
    -   `MIN_RALLY_HITS`: The minimum number of hits required to be considered a highlight-worthy rally.
    -   `MIN_INTERVAL`: The minimum time between hits (to filter out noise).
    -   `MAX_INTERVAL`: The maximum time between hits (a longer interval signifies the end of a point).
-   **Video Slicing:** For each valid rally, it extracts the corresponding video segment.
    -   `PADDING_START` / `PADDING_END`: Adds extra time before the first hit and after the last hit to provide context for the viewer.
-   **Concatenation:** All the extracted clips are joined together to form the final highlight video.

### `calculate_quality_score.py`
A utility for assessing the audio quality of a match. It analyzes the onset strength of detected ball hit candidates to provide a numerical score, which can help determine if a video's audio is clear enough for reliable detection.

### Utility Scripts
-   **`convert_audio.py`**: A simple wrapper for `ffmpeg` to extract a WAV audio file from an MP4 video.
-   **`slice_data.py`**: Chops a long audio file and its corresponding label CSV into smaller, fixed-duration segments for easier processing or training.
-   **`count_hits.py`**: Scans a directory of sliced CSV files and provides statistics on how many hits were detected in each.
-   **`label_converter.py`**: Converts time-based labels (seconds) into frame-based labels (frame index).

## Installation and Usage

### Prerequisites
-   Python 3.8+
-   **FFmpeg**: You must have FFmpeg installed and accessible in your system's PATH. This is required by `moviepy` for video processing and `librosa` for loading audio from video files.

### Setup
1.  **Clone the repository:**
    ```bash
    git clone https://github.com/khanhlong247/pickleball_highlight_extractor.git
    cd pickleball_highlight_extractor
    ```

2.  **Install Python dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

### How to Generate a Highlight Video
1.  **Place your video file** in the project directory. For example, `input_sample_match.mp4`.

2.  **Run the detection script (`generate_data.py`):**
    -   Open `generate_data.py` and update the `FULL_MATCH_PATH` variable to point to your video file.
    -   Run the script from your terminal:
        ```bash
        python generate_data.py
        ```
    -   This will create an output directory (e.g., `pickleball_test12`) containing the audio files and a `metadata_dev` folder with `audio_ball_hits.csv`.

3.  **Run the highlight creation script (`create_highlight.py`):**
    -   Open `create_highlight.py`.
    -   Update `VIDEO_PATH` to your original video file.
    -   Update `MASTER_CSV` to the path of the `audio_ball_hits.csv` generated in the previous step.
    -   Optionally, tweak the rally parameters (`MIN_RALLY_HITS`, `MAX_INTERVAL`, etc.) to customize your highlight.
    -   Run the script:
        ```bash
        python create_highlight.py
        ```
    -   Your final video will be saved in the `result_video/` directory as `highlight.mp4`.
