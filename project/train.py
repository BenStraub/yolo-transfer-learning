from ultralytics import YOLO

# Load the pre-trained YOLOv8s model
model = YOLO('yolov8s.pt')

# Freeze all layers initially
for param in model.model.parameters():
    param.requires_grad = False

# Train the model with frozen layers
data_config = 'D:\WorkSpace\Yolo\RiceLeaf\demo\project\dataset.yaml'  # Path to your dataset configuration file
epochs = 50  # Number of epochs for training
img_size = 640  # Image size for training
batch_size = 16  # Batch size
#device = '0,1,2'  # Use all 3 GPUs (GPU IDs 0, 1, and 2) -> train(device=device)

# Train the model with frozen layers
try:
    model.train(data=data_config, epochs=epochs, imgsz=img_size, batch=batch_size)
except Exception as e:
    print(f"An error occurred during initial training: {e}")

# Unfreeze the last few layers for fine-tuning
for param in list(model.model.parameters())[-4:]:
    param.requires_grad = True  # Unfreeze the last 4 layers

# Optionally, re-load the model to ensure you're starting fresh
# model = YOLO('yolov8s.pt')  # Uncomment if you want to start from scratch

# Fine-tune the model
try:
    model.train(data=data_config, epochs=50, imgsz=img_size, batch=batch_size)
except Exception as e:
    print(f"An error occurred during fine-tuning: {e}")

# Save the trained model
model.save('rice_leaf_disease_detection.pt')

print("Training complete and model saved as 'rice_leaf_disease_detection.pt'")
