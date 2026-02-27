import cv2
import os
import glob
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
import argparse

# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------
DATA_DIR = "DATA"
OUTPUT_DIR = "train"
FRAMES_PER_VIDEO = 15  # Extract roughly 15 evenly spaced frames per video

def get_video_files():
    real_videos = glob.glob(os.path.join(DATA_DIR, "Celeb-real", "*.mp4")) + \
                  glob.glob(os.path.join(DATA_DIR, "YouTube-real", "*.mp4")) 
    fake_videos = glob.glob(os.path.join(DATA_DIR, "Celeb-synthesis", "*.mp4"))
    return real_videos, fake_videos

def extract_frames_from_video(video_path, label):
    """
    Extracts evenly spaced frames from a single video and saves them to the train directory.
    """
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return 0
            
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            cap.release()
            return 0
            
        # Calculate exactly which frames to extract based on FRAMES_PER_VIDEO
        step = max(total_frames // FRAMES_PER_VIDEO, 1)
        
        video_name = os.path.basename(video_path).split('.')[0]
        out_folder = os.path.join(OUTPUT_DIR, label)
        
        frames_extracted = 0
        
        for i in range(FRAMES_PER_VIDEO):
            frame_id = i * step
            if frame_id >= total_frames:
                break
                
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
            ret, frame = cap.read()
            if not ret:
                break
                
            out_filename = os.path.join(out_folder, f"{video_name}_frame{frame_id}.jpg")
            
            # Save the frame
            cv2.imwrite(out_filename, frame)
            frames_extracted += 1
            
        cap.release()
        return frames_extracted
        
    except Exception as e:
        print(f"Error processing {video_path}: {e}")
        return 0

def process_video_wrapper(args):
    """Wrapper for ProcessPoolExecutor"""
    video_path, label = args
    return extract_frames_from_video(video_path, label)

def main(max_workers=None):
    os.makedirs(os.path.join(OUTPUT_DIR, "REAL"), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, "FAKE"), exist_ok=True)
    
    print("Gathering video files...")
    real_videos, fake_videos = get_video_files()
    print(f"Found {len(real_videos)} REAL videos and {len(fake_videos)} FAKE videos.")
    
    tasks = [(vid, "REAL") for vid in real_videos] + [(vid, "FAKE") for vid in fake_videos]
    
    print(f"Extracting {FRAMES_PER_VIDEO} frames per video...")
    total_extracted = 0
    
    # Process videos in parallel for massive speedup
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        for count in tqdm(executor.map(process_video_wrapper, tasks), total=len(tasks), desc="Processing Videos"):
            total_extracted += count
            
    print(f"\nExtraction Complete! Total Images Generated: {total_extracted}")
    print(f"Dataset ready at: {os.path.abspath(OUTPUT_DIR)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=os.cpu_count(), help="Number of parallel processes")
    args = parser.parse_args()
    
    main(max_workers=args.workers)
