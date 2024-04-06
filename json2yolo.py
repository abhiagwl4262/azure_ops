from PIL import Image
import json
import os


def coco_normalize_and_convert(json_file_path, image_dir, output_dir):
    """
    Normalize bounding box coordinates from COCO JSON and convert to individual txt files.

    Parameters:
    - json_file_path: Path to the COCO formatted JSON file.
    - image_dir: Directory where images are stored.
    - output_dir: Output directory for normalized bounding boxes in txt format.
    """
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Load JSON data
    with open(json_file_path, "r") as file:
        coco_data = json.load(file)

    # Map image IDs to file names and dimensions
    image_details = {
        image["id"]: {
            "file_name": image["file_name"],
            "width": image["width"],
            "height": image["height"],
        }
        for image in coco_data["images"]
    }

    # Process annotations, normalizing bounding boxes
    for annotation in coco_data["annotations"]:
        image_id = annotation["image_id"]
        image_info = image_details[image_id]
        img_width, img_height = image_info["width"], image_info["height"]

        # Normalize bounding box [x, y, width, height]
        bbox = annotation["bbox"]
        x_min_n, y_min_n = bbox[0] / img_width, bbox[1] / img_height
        width_n, height_n = bbox[2] / img_width, bbox[3] / img_height

        # Generate output text file per image
        base_filename = os.path.splitext(image_info["file_name"])[0]
        txt_file_path = os.path.join(output_dir, f"{base_filename}.txt")

        # Write normalized bounding box to file (appending each new bbox)
        with open(txt_file_path, "a") as txt_file:
            # Assuming category_id maps directly to a class name; adjust as per your class mapping
            class_id = annotation["category_id"]
            if class_id == 1:
                txt_file.write(
                    f"car,{x_min_n},{y_min_n},{width_n},{height_n}\n"
                )  # Update format as needed
            else:
                txt_file.write(
                    f"person,{x_min_n},{y_min_n},{width_n},{height_n}\n"
                )  # Update format as needed


if __name__ == '__main__':
    # Usage
    json_file_path = "/Users/rishabpal/Downloads/rishabh/coco_annotations/val.json"  # Update to actual JSON file path
    image_dir = "/Users/rishabpal/Downloads/rishabh/valid"  # Folder containing the images
    output_dir = "output/txt_files"  # Output directory for normalized TXT files

    coco_normalize_and_convert(json_file_path, image_dir, output_dir)
