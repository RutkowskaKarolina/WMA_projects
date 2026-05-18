import os
import xml.etree.ElementTree as ET
import shutil

# Input folders
IMG_SRC = "materials"
XML_SRC = "labels/train"

# Output folders
OUT_IMG = "chainsaw_dataset/images/train"
OUT_LABEL = "chainsaw_dataset/labels/train"

# Create output folders if they don't exist
os.makedirs(OUT_IMG, exist_ok=True)
os.makedirs(OUT_LABEL, exist_ok=True)

# Class mapping (e.g., "chainsaw" → class ID 0)
class_name_to_id = {"chainsaw": 0}

# Function to convert one XML annotation to YOLO format
def convert(xml_path, out_txt_path, img_width, img_height):
    tree = ET.parse(xml_path)
    root = tree.getroot()

    yolo_lines = []

    for obj in root.findall("object"):
        cls = obj.find("name").text
        if cls not in class_name_to_id:
            continue
        cls_id = class_name_to_id[cls]

        xml_box = obj.find("bndbox")
        x_min = int(xml_box.find("xmin").text)
        y_min = int(xml_box.find("ymin").text)
        x_max = int(xml_box.find("xmax").text)
        y_max = int(xml_box.find("ymax").text)

        # Convert bounding box to YOLO format: (x_center, y_center, width, height) normalized
        x_center = (x_min + x_max) / 2 / img_width
        y_center = (y_min + y_max) / 2 / img_height
        width = (x_max - x_min) / img_width
        height = (y_max - y_min) / img_height

        yolo_lines.append(f"{cls_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")

    # Save converted annotations to .txt file
    with open(out_txt_path, "w") as f:
        f.write("\n".join(yolo_lines))


# Loop through all XML files and convert them
for xml_file in os.listdir(XML_SRC):
    if not xml_file.endswith(".xml"):
        continue

    base = os.path.splitext(xml_file)[0]
    img_path = os.path.join(IMG_SRC, base + ".jpg")
    xml_path = os.path.join(XML_SRC, xml_file)

    # Read image dimensions from the XML
    tree = ET.parse(xml_path)
    size = tree.find("annotation/size")
    if size is None:
        size = tree.find("size")

    width = int(size.find("width").text)
    height = int(size.find("height").text)

    out_txt = os.path.join(OUT_LABEL, base + ".txt")
    out_img = os.path.join(OUT_IMG, base + ".jpg")

    # Convert annotation and copy the corresponding image
    convert(xml_path, out_txt, width, height)
    shutil.copy2(img_path, out_img)

print("Done! .txt and .jpg files are in chainsaw_dataset/")
