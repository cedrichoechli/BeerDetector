import torchvision.transforms as transforms
import cv2
import numpy
import numpy as np
import os
from coco_names import COCO_INSTANCE_CATEGORY_NAMES as coco_names
from classify import beer_classification
import torch
from PIL import Image




def predict(image, model, device, confidence_threshold=0.8):
    """
    Main function for object detection in given image.
    Filters for bottles and returns the corresponding bounding boxes, the classes, the labels and the score.
    """
    # transform the image to tensor
    transform = transforms.Compose([transforms.ToTensor(),])
    image = transform(image).to(device)
    image = image.unsqueeze(0).type(torch.float32) # add a batch dimension

    print(image.dtype)
    # exit()


    outputs = model(image) # get the predictions on the image

    print("Detection is completed.")
    # print the results individually
    # print(f"BOXES: {outputs[0]['boxes']}")
    # print(f"LABELS: {outputs[0]['labels']}")
    # print(f"SCORES: {outputs[0]['scores']}")
    # get all the predicited class names
    pred_classes = [coco_names[i] for i in outputs[0]['labels'].cpu().numpy()]
    # get score for all the predicted objects
    pred_scores = outputs[0]['scores'].detach().cpu().numpy()
    # get all the predicted bounding boxes
    pred_bboxes = outputs[0]['boxes'].detach().cpu().numpy()

    # filters for bottles
    bottles = []
    for i in pred_classes:
        bottles.append(i == 'bottle')
    bottles = np.array(bottles)
    boxes = pred_bboxes.astype(np.int32)

    # get boxes above the threshold score & which are bottles
    relevant_outputs = (pred_scores >= confidence_threshold) & bottles

    if sum(relevant_outputs) > 0:
        return boxes[relevant_outputs], \
               list(np.array(pred_classes)[relevant_outputs]), \
               outputs[0]['labels'][relevant_outputs], \
               pred_scores[relevant_outputs]
    else:
        return[[], [], [], []]



def draw_boxes(boxes, classes, labels, scores, image):
    """
    Returns the image with bounding boxes including labels, detection scores and classification results
    Classifications below 60% confidence are labeled as unknown
    """
    COLORS = np.random.uniform(0, 255, size=(len(coco_names), 3))
    image_np = cv2.cvtColor(np.asarray(image), cv2.COLOR_BGR2RGB)

    # Create a temporary directory for classification
    temp_dir = "temp_crops"
    os.makedirs(temp_dir, exist_ok=True)

    for i, box in enumerate(boxes):
        color = COLORS[labels[i]]

        # Crop and save temporarily for classification
        crop = image.crop(tuple(box))
        temp_path = f"{temp_dir}/temp_{i}.jpg"
        crop.save(temp_path)

        # Get classification
        class_scores, beer_type = beer_classification(temp_path)
        max_score = float(torch.max(class_scores)) * 100

        # Check confidence threshold
        if max_score < 60:
            beer_type = "Unknown"

        # Draw rectangle
        cv2.rectangle(
            image_np,
            (int(box[0]), int(box[1])),
            (int(box[2]), int(box[3])),
            color, 2
        )

        # Draw detection score
        detection_text = f"Bottle: {numpy.round(scores[i], 3) * 100}%"
        cv2.putText(image_np, detection_text, 
                    (int(box[0]), int(box[1]-25)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, 
                    lineType=cv2.LINE_AA)

        # Draw classification result
        class_text = f"{beer_type}: {numpy.round(max_score, 1)}%" if beer_type != "Unknown" else "Unknown"
        cv2.putText(image_np, class_text, 
                    (int(box[0]), int(box[1]-5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, 
                    lineType=cv2.LINE_AA)

        # Clean up temporary file
        os.remove(temp_path)

    # Clean up temporary directory
    os.rmdir(temp_dir)
    return image_np




def save_cropped_image(image, boxes, directory, filename):
    """
    Saves cropped image for each bounding box to specified directory and returns paths
    """
    saved_paths = []
    for i, box in enumerate(boxes):
        image_cropped = image.crop(tuple(box))
        save_path = f"{directory}/{i}-{filename}"
        image_cropped.save(save_path)
        saved_paths.append(save_path)
    return saved_paths
