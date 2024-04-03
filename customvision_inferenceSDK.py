from azure.cognitiveservices.vision.customvision.training import CustomVisionTrainingClient
from azure.cognitiveservices.vision.customvision.prediction import CustomVisionPredictionClient
from azure.cognitiveservices.vision.customvision.training.models import ImageFileCreateBatch, ImageFileCreateEntry, Region
from msrest.authentication import ApiKeyCredentials
import os, json, time

def add_data(project, tags, image_dir, ann_path):

    print ("Adding images...")
    tag_dict = {}
    for tag in tags:
        tag_dict[tag.name] = tag

    tagged_images_with_regions = []
    f =open(ann_path)
    data = json.load(f)

    category_id_name_dict = {}
    for category in data["categories"]:
        category_id_name_dict[category["id"]] = category["name"]

    img_id_fname_dict = {}
    for img in data["images"]:
        img_id_fname_dict[img["id"]] = img["file_name"]

    for ann in data["annotations"]:
        x,y,w,h = ann["bbox"]
        tag_id = tag_dict[category_id_name_dict[ann["category_id"]]].id
        regions = [Region(tag_id=tag_id, left=x,top=y,width=w,height=h)]
        file_name = img_id_fname_dict[ann["image_id"]]
        with open(os.path.join(image_dir, file_name), mode="rb") as image_contents:
            tagged_images_with_regions.append(ImageFileCreateEntry(name=file_name, contents=image_contents.read(), regions=regions))

    # upload_result = trainer.create_images_from_files(project.id, ImageFileCreateBatch(images=tagged_images_with_regions))
    for i in range(0,len(tagged_images_with_regions),64):
        batch = ImageFileCreateBatch(images=tagged_images_with_regions[i:i+64])
        upload_result = trainer.create_images_from_files(project.id, batch)

        if not upload_result.is_batch_successful: 
            print("Image batch upload failed.")
            for image in upload_result.images:
                print("Image status: ", image.status)
    return
#training creds
training_key = os.environ.get("VISION_TRAINING_KEY")
VISION_TRAINING_ENDPOINT = os.environ.get("VISION_TRAINING_ENDPOINT")
credentials = ApiKeyCredentials(in_headers={"Training-key": training_key})
trainer = CustomVisionTrainingClient(VISION_TRAINING_ENDPOINT, credentials)

# Get existing project
project = trainer.get_project(project_id=os.environ.get("PROJECT_ID"))

# fetch tags
tags = trainer.get_tags(project.id)
if not len(tags):
    car_tag = trainer.create_tag(project.id, "car")
    person_tag = trainer.create_tag(project.id, "person")
    tags = [car_tag, person_tag]

add_data(project, tags, "train", "train_new.json")
# add_data(project, tags, "valid", "valid_new.json")

##Train
print ("Training...")
# iteration = trainer.train_project(project.id) #creates new iteration
iteration = trainer.get_iterations(project.id)[0] #Use last iteration
while (iteration.status != "Completed"):
    iteration = trainer.get_iteration(project.id, iteration.id)
    print ("Training status: " + iteration.status)
    print ("Waiting 10 seconds...")
    time.sleep(10)

prediction_resource_id=os.environ.get("VISION_PREDICTION_RESOURCE_ID")

#The iteration is now trained. Publish it to the project endpoint
try:
    trainer.publish_iteration(project.id, iteration.id, iteration.name, prediction_resource_id)
    print ("Done!")
except Exception as e:
    print(e)

#prediction creds
VISION_PREDICTION_ENDPOINT = os.environ.get("VISION_PREDICTION_ENDPOINT")
prediction_key=os.environ.get("VISION_PREDICTION_KEY")
prediction_credentials = ApiKeyCredentials(in_headers={"Prediction-key": prediction_key})
predictor = CustomVisionPredictionClient(VISION_PREDICTION_ENDPOINT, prediction_credentials)


#inference
with open("valid/Abee1_12_png.rf.3415fd50c0de26c1ad6e3c2fc391f47a.jpg", mode="rb") as test_data:
    results = predictor.detect_image(project.id, iteration.name, test_data)

breakpoint()
# Display the results.
for prediction in results.predictions:
    if prediction.probability > 0.5:
        print("\t" + prediction.tag_name + ": {0:.2f}% bbox.left = {1:.2f}, bbox.top = {2:.2f}, bbox.width = {3:.2f}, bbox.height = {4:.2f}".format(prediction.probability * 100, prediction.bounding_box.left, prediction.bounding_box.top, prediction.bounding_box.width, prediction.bounding_box.height))