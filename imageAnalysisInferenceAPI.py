import http.client, urllib.request, urllib.parse, urllib.error, base64
import os, json
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
        tag = prediction['tags'][0]['name']
        probability = prediction['tags'][0]['confidence']
        bbox = prediction['boundingBox']

        # Convert bounding box coordinates from relative to absolute
        left = bbox['x'] * width
        top = bbox['y'] * height
        right = left + bbox['w'] * width
        bottom = top + bbox['h'] * height

        # Create a rectangle patch
        rect = patches.Rectangle((left, top), right - left, bottom - top, linewidth=1, edgecolor='r', facecolor='none')

        # Add the patch to the plot
        plt.gca().add_patch(rect)

        # Add text label
        plt.text(left, top - 5, f'{tag} ({probability:.2f})', color='r', fontsize=10, backgroundcolor='w')
    
    plt.axis('off')
    plt.show()


headers = {
    # Request headers
    "Content-Type": "application/octet-stream",
    "Ocp-Apim-Subscription-Key": os.environ.get("OCP_APIM_SUBSCRIPTION_KEY")
}

# The path to your image file
image_path = "valid/Abee1_12_png.rf.3415fd50c0de26c1ad6e3c2fc391f47a.jpg"

# Read the image file in binary mode
with open(image_path, "rb") as image_file:
    image_data = image_file.read()

params = urllib.parse.urlencode(
    {
        # Request parameters
        "model-name": "customv3",
    }
)

conf_threshold = 0.5
try:
    conn = http.client.HTTPSConnection(
        "object-detection-v2.cognitiveservices.azure.com"
    )
    conn.request(
        "POST",
        "/computervision/imageanalysis:analyze?api-version=2023-02-01-preview&%s"
        % params,
        image_data,
        headers,
    )
    response = conn.getresponse()
    data = response.read()

    json_str = data.decode("utf-8")
    json_objs = json.loads(json_str)
    detections = []
    for detection in json_objs["customModelResult"]["objectsResult"]["values"]:
        if detection["tags"][0]["confidence"] > conf_threshold:
            detections.append(detection)
    print(detections)
    plot(detections, image_path)
    with open("response.json", "w") as fw:
        fw.write(json.dumps(json_objs, indent=4))
    conn.close()
except Exception as e:
    print("[Errno {0}] {1}".format(e.errno, e.strerror))