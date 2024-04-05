import os
import json
import requests
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def write_predictions(results, img_path, out_path):
    image = Image.open(img_path)
    width, height = image.size

    f = open(out_path, "w")
    for prediction in results:
        tag = prediction['tags'][0]['name']
        probability = prediction['tags'][0]['confidence']
        bbox = prediction['boundingBox']

        # Convert bounding box coordinates from relative to absolute
        x = bbox['x']/float(width)
        y = bbox['y']/float(height)
        w = bbox['w']/float(width)
        h = bbox['h']/float(height)
        pred_line = ",".join([tag, str(probability), str(x), str(y), str(w), str(h)]) + "\n"
        f.write(pred_line)

def plot(results, img_path):
    # Load image from URL
    image = Image.open(img_path)

    # Plot the image
    plt.figure(figsize=(10, 10))
    plt.imshow(image)

    # Get image dimensions
    width, height = image.size

    # Plot each prediction as a bounding box
    for prediction in results:
        tag = prediction["tags"][0]["name"]
        probability = prediction["tags"][0]["confidence"]
        bbox = prediction["boundingBox"]

        # Convert bounding box coordinates from relative to absolute
        left = bbox["x"] * width
        top = bbox["y"] * height
        right = left + bbox["w"] * width
        bottom = top + bbox["h"] * height

        # Create a rectangle patch
        rect = patches.Rectangle(
            (left, top),
            right - left,
            bottom - top,
            linewidth=1,
            edgecolor="r",
            facecolor="none",
        )

        # Add the patch to the plot
        plt.gca().add_patch(rect)

        # Add text label
        plt.text(
            left,
            top - 5,
            f"{tag} ({probability:.2f})",
            color="r",
            fontsize=10,
            backgroundcolor="w",
        )

    plt.axis("off")
    plt.show()


def infer(image_path, conf_threshold=0.5, plot=False, out_path="output.txt"):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"{image_path} does not exist!")

    with open(image_path, "rb") as image_file:
        image_data = image_file.read()

    headers = {
        "Content-Type": "application/octet-stream",
        "Ocp-Apim-Subscription-Key": subscription_key,
    }
    params = {"model-name": "customv3", "api-version": "2023-02-01-preview"}
    url = "https://object-detection-v2.cognitiveservices.azure.com/computervision/imageanalysis:analyze"

    try:
        response = requests.post(
            url,
            params=params,
            headers=headers,
            data=image_data,
        )
        response.raise_for_status
        json_objs = response.json()
        detections = []
        for detection in json_objs["customModelResult"]["objectsResult"]["values"]:
            if detection["tags"][0]["confidence"] > conf_threshold:
                detections.append(detection)
        print(detections)
        if plot:
            plot(detections, image_path)
        write_predictions(detections, image_path, out_path)
    except requests.exceptions.RequestException as e:
        print("Request failed: ", e)


if __name__ == "__main__":
    subscription_key = os.environ.get("OCP_APIM_SUBSCRIPTION_KEY")
    assert (
        subscription_key
    ), "OCP_APIM_SUBSCRIPTION_KEY environment variable is not set."

    conf_threshold = 0.5
    # image_path = "Image 2024-03-26 at 09.56.02.jpeg"
    image_path = "Image 2024-03-26 at 22.08.49.jpeg"

    infer(image_path, conf_threshold)
