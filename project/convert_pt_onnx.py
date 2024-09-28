import torch
from ultralytics import YOLO

# Input the torch model name that are gonna convert to
model = YOLO('rice_leaf_disease_detection_freeze.pt')

img_size = 640

# Now, let's convert the PyTorch model to ONNX
def export_to_onnx(model, onnx_model_path='rice_leaf_disease_detection.onnx'):
    # Example dummy input for the model (matching the image size you used)
    dummy_input = torch.randn(1, 3, img_size, img_size).to(model.device)

    # Export the model to ONNX format
    try:
        torch.onnx.export(
            model.model,  # The PyTorch model
            dummy_input,  # Example input tensor to the model
            onnx_model_path,  # Path where the ONNX model will be saved
            opset_version=11,  # ONNX version to export to (11 is commonly used)
            input_names=['input'],  # Input layer name
            output_names=['output'],  # Output layer name
            dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}  # Dynamic batch size
        )
        print(f"Model has been successfully exported to ONNX format at '{onnx_model_path}'")
    except Exception as e:
        print(f"Error occurred while exporting to ONNX: {e}")

# Call the export function
export_to_onnx(model, 'rice_leaf_disease_detection.onnx')