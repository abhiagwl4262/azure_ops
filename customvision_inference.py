from azure.cognitiveservices.vision.customvision.training import CustomVisionTrainingClient
from azure.cognitiveservices.vision.customvision.prediction import CustomVisionPredictionClient
from msrest.authentication import ApiKeyCredentials
import os

#training creds
training_key = os.environ("VISION_TRAINING_KEY")
VISION_TRAINING_ENDPOINT = os.environ("VISION_TRAINING_ENDPOINT")
credentials = ApiKeyCredentials(in_headers={"Training-key": training_key})
trainer = CustomVisionTrainingClient(VISION_TRAINING_ENDPOINT, credentials)

# Get existing project
project = trainer.get_project(project_id=os.environ("PROJECT_ID"))

#prediction creds
VISION_PREDICTION_ENDPOINT = os.environ("VISION_PREDICTION_ENDPOINT")
prediction_key=os.environ("VISION_PREDICTION_KEY")
prediction_resource_id=os.environ("VISION_PREDICTION_RESOURCE_ID")

prediction_credentials = ApiKeyCredentials(in_headers={"Prediction-key": prediction_key})
predictor = CustomVisionPredictionClient(VISION_PREDICTION_ENDPOINT, prediction_credentials)


#inference
publish_iteration_name="Iteration1"
with open("valid/Abee1_12_png.rf.3415fd50c0de26c1ad6e3c2fc391f47a.jpg", mode="rb") as test_data:
    results = predictor.detect_image(project.id, publish_iteration_name, test_data)

# Display the results.    
for prediction in results.predictions:
    print("\t" + prediction.tag_name + ": {0:.2f}% bbox.left = {1:.2f}, bbox.top = {2:.2f}, bbox.width = {3:.2f}, bbox.height = {4:.2f}".format(prediction.probability * 100, prediction.bounding_box.left, prediction.bounding_box.top, prediction.bounding_box.width, prediction.bounding_box.height))