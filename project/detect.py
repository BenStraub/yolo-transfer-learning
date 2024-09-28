import cv2
from ultralytics import YOLO
import matplotlib.pyplot as plt
import os

def main(model_path='rice_leaf_disease_detection.pt'):
    # Load the trained YOLOv8 model
    model = YOLO(model_path)

    # Open a connection to the webcam
    cap = cv2.VideoCapture(0)  # 0 is the default camera

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return
    
    # Create a directory to save images if it doesn't exist
    save_dir = 'saved_images'
    os.makedirs(save_dir, exist_ok=True)

    frame_count = 0  # To keep track of the frame number for naming images
    
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break

        # Make predictions
        results = model.predict(source=frame, show=False)

        # Process results
        for result in results:
            # Get the bounding boxes, class names, and confidence scores
            boxes = result.boxes
            for box in boxes:
                # Get box coordinates, class id, and score
                x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box coordinates
                conf = box.conf[0]  # Confidence score
                class_id = int(box.cls[0])  # Class index
                class_name = result.names[class_id]  # Class name

                # Draw a red bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), color=(0, 0, 255), thickness=2)  # Red color

                # Display the class name and confidence above the bounding box
                label = f"{class_name}: {conf:.2f}"
                label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
                cv2.rectangle(frame, (x1, y1 - label_size[1] - 5), (x1 + label_size[0], y1), (0, 0, 255), -1)
                cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        # Display the resulting frame
        cv2.imshow('Real-Time Object Detection', frame)
        
        # Save the frame to a file every 10 frames
        if frame_count % 2 == 0:
            filename = os.path.join(save_dir, f"frame_{frame_count}.jpg")
            cv2.imwrite(filename, frame)
            print(f"Saved: {filename}")

        frame_count += 1

        # Exit loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the capture and close windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
