import os
import torchvision
import torch
import argparse
import cv2
import detect_utils
import numpy
import numpy as np
from PIL import Image
from classify import beer_classification


def detect(image_location, save_cropped_image=False,  output_filename='outputs/test.jpg', show_image=False, save_image=False):
    """
    beer detection on given image
    options to show the image with bounding boxes and crop the content of bounding box to seperate file
    """
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    device = torch.device('cpu')
    image = Image.open(image_location)

    # object detection on image with confidence_threshold
    model.eval().to(device)
    confidence_threshold = 0.8
    boxes, classes, labels, scores = detect_utils.predict(image, model, device, confidence_threshold)

    if save_cropped_image:
        # save cropped image for each bounding box to outputs folder
        directory = os.path.dirname(output_filename)
        filename = os.path.basename(output_filename)
        detect_utils.save_cropped_image(image, boxes, directory, filename)

    if show_image or save_image:
        # draw bounding boxes with detection and classification results
        image = detect_utils.draw_boxes(boxes, classes, labels, scores, image)

        if show_image:
            cv2.imshow('Image', image)
            cv2.waitKey(0)

        if save_image:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            Image.fromarray(image_rgb).save(str(output_filename))



if __name__ == '__main__':
    # construct argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', help='path to input image')
    args = vars(parser.parse_args())

    detect(args['input'], save_cropped_image=True, show_image=False, save_image=True)
