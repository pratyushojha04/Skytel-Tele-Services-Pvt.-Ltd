# Vehicle Detection and Tracking System

## Overview
This project implements a vehicle detection and tracking system using the YOLOv8 model and the SORT (Simple Online and Realtime Tracking) algorithm. It processes a video to detect vehicles, track them across frames, and count vehicles passing through predefined lanes, saving the results to a CSV file and an output video with annotations.

## Features
- **Vehicle Detection**: Uses YOLOv8 (`yolov8n.pt`) to detect vehicles (cars, motorcycles, buses, trucks) in video frames.
- **Object Tracking**: Implements the SORT algorithm with Kalman filtering to track vehicles across frames.
- **Lane Counting**: Divides the video frame into three lanes and counts vehicles crossing a horizontal line at 60% of the frame height.
- **Output**: 
  - Saves tracking results (vehicle ID, lane number, frame count, timestamp) to `vehicle_counts.csv`.
  - Generates an annotated output video (`output_video.mp4`) with bounding boxes, vehicle IDs, lane boundaries, and vehicle counts per lane.
- **Video Source**: Downloads a video from a YouTube URL using `yt-dlp` if not already present.

## Requirements
To run this project, install the following Python packages:
```bash
pip install opencv-python numpy ultralytics yt-dlp filterpy scipy
```

## Dependencies
- **Python**: 3.6 or higher
- **OpenCV**: For video processing and visualization
- **NumPy**: For numerical operations
- **Ultralytics YOLO**: For object detection
- **yt-dlp**: For downloading YouTube videos
- **FilterPy**: For Kalman filtering in SORT
- **SciPy**: For Hungarian algorithm in tracking

## Usage
1. **Prepare the Video**:
   - The script automatically downloads a video from the specified YouTube URL (`https://www.youtube.com/watch?v=MNn9qKG2UFI`) and saves it as `traffic_video.mp4`.
   - If the download fails, manually download the video and place it in the project directory as `traffic_video.mp4`.

2. **Run the Script**:
   ```bash
   python vehicle_tracking.py
   ```
   Replace `vehicle_tracking.py` with the name of your script file.

3. **Output**:
   - **Video**: An annotated video (`output_video.mp4`) showing detected vehicles with bounding boxes, vehicle IDs, lane boundaries, and vehicle counts per lane.
   - **CSV File**: `vehicle_counts.csv` containing columns: `Vehicle ID`, `Lane number`, `Frame count`, `Timestamp`.
   - **Console Output**: Summary of total vehicles counted per lane.

## How It Works
1. **Video Download**:
   - Uses `yt-dlp` to download the video if not already present.
2. **Vehicle Detection**:
   - YOLOv8 detects vehicles (classes: car, motorcycle, bus, truck) with a confidence threshold of 0.5.
3. **Vehicle Tracking**:
   - The SORT algorithm assigns unique IDs to vehicles and tracks them using Kalman filtering and IoU-based association.
4. **Lane Counting**:
   - The frame is divided into three lanes using vertical lines at 1/3 and 2/3 of the frame width.
   - A vehicle is counted when its centroid crosses a horizontal line at 60% of the frame height, assigned to a lane based on its x-coordinate.
5. **Visualization**:
   - Bounding boxes and vehicle IDs are drawn on each detected vehicle.
   - Lane boundaries (blue lines) and the counting line (red line) are overlaid on the video.
   - Vehicle counts for each lane are displayed at the top of the frame.
6. **Data Logging**:
   - Each counted vehicle’s details (ID, lane, frame, timestamp) are logged to `vehicle_counts.csv`.

## Configuration
- **Video URL**: Modify `video_url` in the script to process a different YouTube video.
- **Counting Line**: Adjust `counting_line_y = height * 0.6` to change the position of the counting line.
- **Lane Boundaries**: Modify the lane division logic (currently at `width / 3` and `2 * width / 3`) for different lane configurations.
- **Vehicle Classes**: Update `vehicle_classes = [2, 3, 5, 7]` to include other YOLO class IDs if needed.
- **SORT Parameters**:
   - `max_age`: Maximum frames a tracker persists without updates (default: 20).
   - `min_hits`: Minimum detections before a track is confirmed (default: 3).
   - `iou_threshold`: IoU threshold for matching detections to trackers (default: 0.3).

## File Structure
```
project_directory/
├── vehicle_tracking.py       # Main script
├── traffic_video.mp4         # Input video (downloaded or provided)
├── output_video.mp4          # Output annotated video
├── vehicle_counts.csv        # Output CSV with vehicle counts
├── yolov8n.pt                # YOLOv8 model weights (downloaded by Ultralytics)
├── README.md                 # This file
```

## Notes
- Ensure a stable internet connection for downloading the video.
- The YOLOv8 model (`yolov8n.pt`) is automatically downloaded by the Ultralytics library on first run.
- The script assumes the video has a clear view of vehicles moving in lanes.
- For better performance, use a more powerful YOLO model (e.g., `yolov8m.pt`) or adjust the confidence threshold.

## Limitations
- The lane assignment is based on simple x-coordinate division, which may not work for complex road layouts.
- The script assumes vehicles move downward across the counting line.
- Performance depends on the quality of the input video and the YOLO model’s accuracy.

## Future Improvements
- Add support for multiple counting lines or dynamic lane detection.
- Implement direction detection to handle vehicles moving in any direction.
- Enhance visualization with additional annotations (e.g., vehicle type).
- Optimize for real-time processing on high-resolution videos.

## License
This project is licensed under the MIT License.