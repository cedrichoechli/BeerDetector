import torchvision.transforms as transforms
import cv2
import numpy
import numpy as np
from coco_names import COCO_INSTANCE_CATEGORY_NAMES as coco_names





def predict(image, model, device, confidence_threshold=0.8):
    """
    Main function for object detection in given image.
    Filters for bottles and returns the corresponding bounding boxes, the classes, the labels and the score.
    """
    # transform the image to tensor
    transform = transforms.Compose([transforms.ToTensor(),])
    image = transform(image).to(device)
    image = image.unsqueeze(0) # add a batch dimension
    outputs = model(image) # get the predictions on the image
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
    Returns the image with bounding boxes including labels and scores
    """
    COLORS = np.random.uniform(0, 255, size=(len(coco_names), 3))
    image = cv2.cvtColor(np.asarray(image), cv2.COLOR_BGR2RGB)
    for i, box in enumerate(boxes):
        color = COLORS[labels[i]]
        # draw rectangle with dimensions specified in boxes
        cv2.rectangle(
            image,
            (int(box[0]), int(box[1])),
            (int(box[2]), int(box[3])),
            color, 2
        )
        # put the labels with the predicted scores above the rectangle
        cv2.putText(image, classes[i] + " : " + str(numpy.round(scores[i], 3) * 100) + "%", (int(box[0]), int(box[1]-5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, 
                    lineType=cv2.LINE_AA)
    return image




def save_cropped_image(image, boxes, directory, filename):
    """
    Saves cropped image for each bounding box to specified directory
    """
    file_id = 0
    for box in boxes:
        image_cropped = image.crop(tuple(box))
        image_cropped.save(f"{directory}/{str(file_id)}-{filename}")
        file_id += 1
