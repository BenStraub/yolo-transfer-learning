# Rice Leaf Health Status Classification System

This is a demo project of the Measure System of the Health Status of the Rice Leaf using YOLOv8

## Methodolgy

### Step 1. Data Collection

- Collect Images:
  Use your phone camera to capture images of rice leaves in various conditions (healthy, diseased, etc.).
  Ensure diversity in lighting, angles, and backgrounds to improve the model's robustness.

- Label the Images:
  Use a labeling tool like LabelImg or Roboflow to annotate your images.
  Create bounding boxes around the leaves and classify their status (e.g., healthy, diseased).

- Organize the Data:
  Save the images and their corresponding annotation files in a structured format (e.g., a folder for images and another for annotations).
  
### Step 2. Prepare the Dataset

- Split the Dataset:
  Divide your dataset into training, validation, and testing sets. A common split is 70% training, 20% validation, and 10% testing.

- Convert Annotations:  
  Ensure your annotations are in YOLO format (i.e., a .txt file for each image with class and bounding box coordinates). The format is:

  ``` bash
  class_id x_center y_center width height
  ```
  Normalize the bounding box coordinates (values between 0 and 1).
  
  If the result of the annotation with LabelImg is VOC format, please use the yolo_format.py
  
- Organize the Dataset
  Once the images are labeled, organize your dataset in the following structure, which YOLO requires:

  ``` bash
  dataset/
  ├── images/
  │   ├── train/  # Training images
  │   └── val/    # Validation images
  └── labels/
      ├── train/  # Corresponding labels for training images
      └── val/    # Corresponding labels for validation images
  ```  
  
- Data Augmentation
  referrence the data_aug.py file

### Step 3. Train the YOLOv8 model

1. Set up Environment
- Install the necessary packages. Make sure you have Python, PyTorch, and YOLOv8 installed. You can install YOLOv8 using:
```bash
pip install ultralytics
```

2. Prepare the Training Script
- Create a YAML configuration file for your dataset. Here's an example (data.yaml):
```bash
train: path/to/train/images
val: path/to/val/images

nc: 2  # number of classes
names: ['healthy', 'diseased']  # class names
```

3. Train the model
- Use the following command to train your YOLOv8 model:
```bash
yolov8 train --data data.yaml --weights yolov8s.pt --epochs 50 --img 640
```

- Adjust the --weights parameter if you want to use a different YOLOv8 model (e.g., yolov8m.pt for a medium model).

4. Evaludate the Model
- After training, evaluate your model's performance on the validation set. Use the following command to test:

```bash
yolov8 val --data data.yaml --weights runs/train/exp/weights/best.pt
```


## Package URL

https://drive.google.com/drive/folders/1oukfS6Iob8k9dAA_DyxG2fHgJrATCjo_?usp=sharing