import cv2
import os

# Open the AVI file
video_path = 'WIN_20240928_04_39_59_Pro.mp4'  # Path to your AVI file
output_dir = 'frames_output'  # Directory to save frames

# Create output directory if it doesn't exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Open video file
cap = cv2.VideoCapture(video_path)

# Check if video opened successfully
if not cap.isOpened():
    print("Error: Could not open video file.")
    exit()

frame_count = 0

# Loop through video frames
while True:
    ret, frame = cap.read()  # Read the next frame
    if not ret:
        break  # Break the loop when no more frames are available

    # Save frame as PNG or JPG file
    output_path = os.path.join(output_dir, f"frame_{frame_count:04d}.png")  # Change to .jpg for JPG
    cv2.imwrite(output_path, frame)
    
    print(f"Saved {output_path}")
    frame_count += 1

# Release video capture object
cap.release()

print(f"Total frames extracted: {frame_count}")
