import requests
import json

prediction_url = "https://customobjectdetection-prediction.cognitiveservices.azure.com/customvision/v3.0/Prediction/8f29a6c3-5b39-4999-92b6-e5d9f3573ec9/detect/iterations/Iteration%204/image"

headers = {
    # Request headers
    "Content-Type": "application/octet-stream",
    "Prediction-Key": "5deb696fc78948469145457b4d52fe11"
}
# The path to your image file
image_path = "valid/Abee1_12_png.rf.3415fd50c0de26c1ad6e3c2fc391f47a.jpg"
# Read the image file in binary mode
with open(image_path, "rb") as image_file:
    data = image_file.read()

try:
    response = requests.post(prediction_url, data=data, headers=headers)
    if response.status_code == 200:
        # Parse and print the prediction results
        results = json.loads(response.content.decode('utf-8'))
        for prediction in results['predictions']:
            print(f"Tag: {prediction['tagName']}, Probability: {prediction['probability']}")
    else:
        print(f"Prediction failed with status code {response.status_code}: {response.text}")
except Exception as e:
    print("[Errno {0}] {1}".format(e.errno, e.strerror))