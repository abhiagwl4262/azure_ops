import http.client, urllib.request, urllib.parse, urllib.error, base64
import os, json

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
    with open("response.json", "w") as fw:
        fw.write(json.dumps(json_objs, indent=4))
    print(json.dumps(json_objs, indent=4))
    conn.close()
except Exception as e:
    print("[Errno {0}] {1}".format(e.errno, e.strerror))