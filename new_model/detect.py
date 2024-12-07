import torchvision
import numpy
import torch
import argparse
import cv2
import detect_utils
from PIL import Image


if __name__ == '__main__':
    # construct argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', help='path to input image')
    args = vars(parser.parse_args())


    
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    image = Image.open(args['input'])

    # object detection on image with confidence_threshold
    model.eval().to(device)
    confidence_threshold = 0.8
    boxes, classes, labels, scores = detect_utils.predict(image, model, device, confidence_threshold)

    # save cropped image for each bounding box to outputs folder
    # detect_utils.save_cropped_image(image, boxes)

    # draw bounding boxes
    image = detect_utils.draw_boxes(boxes, classes, labels, scores, image)

    # show image
    cv2.imshow('Image', image)
    cv2.waitKey(0)
