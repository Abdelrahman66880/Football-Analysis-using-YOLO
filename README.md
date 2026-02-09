# ⚽ Football Analysis using YOLO

A comprehensive computer vision project that performs real-time analysis of football (soccer) matches using YOLOv8 object detection. This system automatically detects players, referees, and the ball, assigns players to teams, tracks their movements, and generates detailed analytics including speed and distance estimation.

## 🌟 Features

- **Real-time Object Detection**: Detect players, referees, and the ball using YOLOv8
- **Player Tracking**: Track individual players across video frames using ByteTrack
- **Team Assignment**: Automatically assign players to teams based on jersey color clustering
- **Ball Possession Tracking**: Determine which player has possession of the ball
- **Speed & Distance Estimation**: Calculate player speeds and distances covered
- **Perspective Transformation**: Convert pixel coordinates to real-world field coordinates
- **Annotated Video Output**: Generate output videos with visual analytics overlays
- **Stub Caching**: Save tracking data for faster processing and analysis

## 📋 Prerequisites

- Python 3.8+
- OpenCV (`cv2`)
- PyTorch
- Ultralytics YOLO
- Scikit-learn
- Pandas
- NumPy
- Supervision library (for tracking)

## 🚀 Installation

### 1. Clone the Repository
```bash
git clone https://github.com/Abdelrahman66880/Football-Analysis-using-YOLO.git
cd Football-Analysis-using-YOLO
```

### 2. Create a Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### Required Packages
```
ultralytics>=8.0.0
opencv-python>=4.5.0
torch>=1.9.0
torchvision>=0.10.0
scikit-learn>=0.24.0
pandas>=1.3.0
numpy>=1.21.0
supervision>=0.3.0
```

## 📁 Project Structure

```
Football-Analysis-using-YOLO/
├── main.py                          # Main pipeline execution
├── yolo_inference.py                # YOLO model inference examples
├── models/
│   ├── best.pt                      # Best trained YOLO model
│   └── last.pt                      # Last checkpoint
├── input_videos/                    # Input video directory
│   └── Video_1.mp4                  # Sample input video
├── output_videos/                   # Generated annotated videos
├── stubs/                           # Cached tracking data
│   └── track_stubs.pkl              # Serialized track data
├── utils/
│   ├── bbox_utils.py                # Bounding box utilities
│   └── video_utils.py               # Video processing utilities
├── trackers/
│   └── tracker.py                   # Object detection & tracking
├── team_assigner/
│   └── team_assigner.py             # Team color assignment
├── player_ball_assigner/
│   └── player_ball_assigner.py      # Ball possession detection
├── speed_and_distance_estimator/
│   └── speed_and_distance_estimator.py  # Performance metrics
├── view_transformer/
│   └── view_transformer.py          # Perspective transformation
├── development_and_analysis/
│   └── color_assignment.ipynb       # Development notebooks
└── train/
    └── football_training_yolo_v5.ipynb  # Model training notebook
```

## 🎯 Quick Start

### Basic Usage

1. **Place your video** in the `input_videos/` directory
2. **Update the video path** in `main.py` if needed
3. **Run the pipeline**:
   ```bash
   python main.py
   ```
4. **View results** in `output_videos/output_video.avi`

### Example

```python
from main import main

# Run the complete analysis pipeline
main()
```

## 🔧 Module Description

### 1. **Tracker** (`trackers/tracker.py`)
Handles YOLO model inference and ByteTrack tracking.

**Key Functions:**
- `get_object_tracks()`: Detect and track objects across frames
- `interpolate_ball_positions()`: Fill missing ball detections
- `add_position_to_tracks()`: Calculate object positions
- `draw_annotations()`: Create visualized output

### 2. **Team Assigner** (`team_assigner/team_assigner.py`)
Automatically assign players to teams using K-Means clustering on jersey colors.

**Key Functions:**
- `assign_team_color()`: Identify team colors from first frame
- `get_player_color()`: Extract dominant color from player jersey
- `get_player_team()`: Assign team ID to each player

### 3. **Player-Ball Assigner** (`player_ball_assigner/player_ball_assigner.py`)
Determine which player is closest to the ball and likely has possession.

**Key Functions:**
- `assign_ball_to_player()`: Find nearest player to ball within threshold distance

### 4. **Speed & Distance Estimator** (`speed_and_distance_estimator/speed_and_distance_estimator.py`)
Calculate player performance metrics based on position changes across frames.

**Key Functions:**
- `add_speed_and_distance_to_tracks()`: Compute speed and distance
- `draw_speed_and_distance()`: Overlay metrics on video

### 5. **View Transformer** (`view_transformer/view_transformer.py`)
Transform pixel coordinates to real-world field coordinates using perspective transformation.

**Key Functions:**
- `transform_point()`: Convert pixel position to field position
- `add_transformed_position_to_tracks()`: Transform all tracked positions

### 6. **Utilities** (`utils/`)
Helper functions for common operations.

**Functions:**
- `get_center_of_bbox()`: Extract center coordinates from bounding box
- `get_foot_position()`: Get foot position for players
- `measure_distance()`: Calculate Euclidean distance
- `read_video()`: Load video frames
- `save_video()`: Write output video

## 📊 Pipeline Workflow

```
Input Video
    ↓
YOLO Detection (Players, Referees, Ball)
    ↓
ByteTrack Tracking
    ↓
Team Color Assignment (K-Means Clustering)
    ↓
Ball Position Interpolation
    ↓
Calculate Positions (feet, center)
    ↓
Perspective Transformation
    ↓
Speed & Distance Estimation
    ↓
Ball Possession Assignment
    ↓
Draw Annotations
    ↓
Output Video + Team Ball Control Stats
```

## ⚙️ Configuration

### Adjustable Parameters

**Player-Ball Distance Threshold** (`player_ball_assigner.py`):
```python
self.max_player_ball_distance = 70  # Pixels
```

**View Transformer Court Dimensions** (`view_transformer.py`):
```python
court_width = 68  # meters
court_length = 23.32  # meters
```

**Video Processing** (`video_utils.py`):
```python
fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Codec
fps = 24  # Frames per second
```

## 🎬 Output

The system generates:
- **Annotated Video**: Output video with overlays showing:
  - Player bounding boxes with IDs
  - Team assignments (color-coded)
  - Ball position and possession
  - Speed indicators
  - Team ball control statistics
- **Track Stubs**: Cached pickle files for faster re-processing

## 📈 Expected Output Format

Each frame includes:
- Player positions with unique IDs
- Team colors (color-coded ellipses)
- Ball position marker
- Speed indicators for each player
- Team ball control percentages

## 🐛 Troubleshooting

### Video Not Processing
- Ensure video format is supported (MP4, AVI, MOV)
- Check video path in `main.py`
- Verify input video exists

### Low Detection Accuracy
- Update YOLO model weights (`models/best.pt`)
- Adjust detection confidence threshold
- Ensure adequate lighting in video

### Memory Issues
- Reduce batch size in `tracker.py`
- Process shorter video clips
- Use lower resolution videos

### Missing Ball Detections
- Enable ball position interpolation (already done in pipeline)
- Check if ball is in frame
- Verify YOLO model was trained on football data

## 🔄 Training Custom Models

Train your own YOLO model on custom football footage:

```bash
# See train/football_training_yolo_v5.ipynb for detailed training notebook
```

## 🚧 Future Enhancements

- [ ] Real-time streaming support
- [ ] Multi-camera tracking
- [ ] Advanced player action recognition
- [ ] Pass accuracy analytics
- [ ] Heatmap generation
- [ ] Web interface
- [ ] Mobile app integration

## 📚 References

- [YOLOv8 Documentation](https://docs.ultralytics.com/)
- [ByteTrack](https://github.com/ifzhang/ByteTrack)
- [Supervision Library](https://github.com/roboflow-ai/supervision)
- [OpenCV Documentation](https://docs.opencv.org/)

## 📝 License

This project is open source and available under the MIT License.

## 👤 Author

**Your Name**
- GitHub: [@yourusername](https://github.com/yourusername)
- Email: your.email@example.com

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

### Steps to Contribute:
1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## ⭐ Acknowledgments

- Thanks to the Ultralytics team for YOLOv8
- ByteTrack for excellent multi-object tracking
- Supervision library for computer vision utilities

---

**Last Updated**: February 10, 2026

For issues and questions, please open an issue on GitHub.
