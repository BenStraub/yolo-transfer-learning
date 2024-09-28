import cv2
import numpy as np
import onnxruntime as ort
import os

def preprocess(image):
    # Resize and normalize the image for the model input
    img = cv2.resize(image, (640, 640))  # Resize to model input size
    img = img.astype(np.float32) / 255.0  # Normalize to [0, 1]
    img = np.transpose(img, (2, 0, 1))  # Change data format from HWC to CHW
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img

def main(model_path='rice_leaf_disease_detection.onnx'):
    # Load the ONNX model
    session = ort.InferenceSession(model_path)

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

        # Preprocess the frame
        input_tensor = preprocess(frame)

        # Make predictions
        results = session.run(None, {session.get_inputs()[0].name: input_tensor})
        print("Results shape:", [result.shape for result in results])
        

        # Process results from the first output (bounding boxes)
        boxes = results[0]  # (1, 6, 8400)
        boxes = boxes[0]  # Take the first batch

        # Filter the boxes based on confidence
        for i in range(boxes.shape[0]):
            box = boxes[i]
            # Assuming box format is [x1, y1, x2, y2, obj_conf, class_conf]
            obj_conf = box[4]  # Object confidence
            if obj_conf > 0.5:  # Confidence threshold
                x1, y1, x2, y2 = map(int, box[:4])  # Get bounding box coordinates
                class_conf = box[5]  # Class confidence
                class_id = np.argmax(box[5:])  # Class index based on the max confidence

                # Draw a red bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), color=(0, 0, 255), thickness=2)

                # Display the class name and confidence above the bounding box
                label = f"Class {class_id}: {obj_conf:.2f}"  # Modify this to match your class names
                label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
                cv2.rectangle(frame, (x1, y1 - label_size[1] - 5), (x1 + label_size[0], y1), (0, 0, 255), -1)
                cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        # Display the resulting frame
        cv2.imshow('Real-Time Object Detection', frame)
        
        # Save the frame to a file every 10 frames
        if frame_count % 10 == 0:
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
