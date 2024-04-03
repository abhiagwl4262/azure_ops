import requests
import json
import os
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches

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
        tag = prediction["tagName"]
        probability = prediction["probability"]
        bbox = prediction["boundingBox"]

        # Convert bounding box coordinates from relative to absolute
        left = bbox["left"] * width
        top = bbox["top"] * height
        right = left + bbox["width"] * width
        bottom = top + bbox["height"] * height

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


def infer(image_path, conf_threshold=0.5):
    prediction_url = "https://customobjectdetection-prediction.cognitiveservices.azure.com/customvision/v3.0/Prediction/8f29a6c3-5b39-4999-92b6-e5d9f3573ec9/detect/iterations/Iteration%204/image"

    headers = {
        # Request headers
        "Content-Type": "application/octet-stream",
        "Prediction-Key": os.environ.get("VISION_PREDICTION_KEY"),
    }

    # Read the image file in binary mode
    with open(image_path, "rb") as image_file:
        data = image_file.read()

    try:
        response = requests.post(prediction_url, data=data, headers=headers)
        if response.status_code == 200:
            # Parse and print the prediction results
            results = json.loads(response.content.decode("utf-8"))
            filtered_preds = []
            for i, prediction in enumerate(results["predictions"]):
                if prediction["probability"] >= conf_threshold:
                    filtered_preds.append(prediction)
                    print(
                        f"Tag: {prediction['tagName']}, Probability: {prediction['probability']}"
                    )
            # plot(filtered_preds, image_path)
        else:
            print(
                f"Prediction failed with status code {response.status_code}: {response.text}"
            )
    except Exception as e:
        print("[Errno {0}] {1}".format(e.errno, e.strerror))


if __name__ == "__main__":
    conf_threshold = 0.5
    image_path = "Image 2024-03-26 at 09.56.02.jpeg"

    infer(image_path, conf_threshold)
