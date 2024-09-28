import os
import xml.etree.ElementTree as ET

def convert(size, box):
    dw = 1. / size[0]
    dh = 1. / size[1]
    x = (box[0] + box[1]) / 2.0 - 1
    y = (box[2] + box[3]) / 2.0 - 1
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return (x, y, w, h)

def convert_annotation(xml_file, class_mapping):
    tree = ET.parse(xml_file)
    root = tree.getroot()

    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)

    output_lines = []
    
    for obj in root.iter('object'):
        difficult = obj.find('difficult').text
        cls = obj.find('name').text
        if cls not in class_mapping or int(difficult) == 1:
            continue
        
        cls_id = class_mapping[cls]
        xml_box = obj.find('bndbox')
        b = (float(xml_box.find('xmin').text), float(xml_box.find('xmax').text),
             float(xml_box.find('ymin').text), float(xml_box.find('ymax').text))
        bb = convert((w, h), b)
        output_lines.append(f"{cls_id} {' '.join(map(str, bb))}\n")

    return output_lines

def process_annotations(input_dir, output_dir, class_mapping):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for filename in os.listdir(input_dir):
        if filename.endswith('.xml'):
            xml_path = os.path.join(input_dir, filename)
            txt_filename = filename.replace('.xml', '.txt')
            txt_path = os.path.join(output_dir, txt_filename)
            
            with open(txt_path, 'w') as txt_file:
                txt_file.writelines(convert_annotation(xml_path, class_mapping))

if __name__ == "__main__":
    # Define your class to ID mapping here
    class_mapping = {
        'caffee': 0,
        'buss': 1,
        'cat': 2
        # Add more classes if needed
    }
    
    input_dir = 'input'  # Replace with your directory containing XML files
    output_dir = 'output'  # Replace with your target directory for YOLO format .txt files

    process_annotations(input_dir, output_dir, class_mapping)
