import torch
import torchvision.models as models
from torchvision import transforms
from PIL import Image

import numpy as np

from torchvision.models import wide_resnet50_2

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import os


model = wide_resnet50_2(pretrained=True)

def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    image = Image.open(image_path)
    raw_image = np.array(image)
    image = transform(image)
    image = image.unsqueeze(0)

    return image, raw_image


def predict_image(image_path):
    model.eval()

    input_image, raw_image = preprocess_image(image_path)

    with torch.no_grad():
        output = model(input_image)

    # get topn elements from output tensor
    topn = 5
    probs = torch.nn.functional.softmax(output, dim=1)
    values, predicted = torch.topk(probs, topn)
  
    # get classname from ID
    with open('classes.txt') as file:
        lines = file.readlines()

    # print predicted items and their corresponding class name
    for i in predicted[0]:
        print(f'{i} - {lines[i]}')

    # create visualization
    plt.figure(figsize=(10,5))

    # Plot image
    plt.subplot(1,2,1)
    plt.imshow(raw_image)
    plt.axis('off')
    plt.title('Input Image')

    # Plot predictions
    plt.subplot(1,2,2)
    y_pos = np.arange(topn)
    percentages = values[0].numpy() * 100
    plt.barh(y_pos, percentages)
    plt.yticks(y_pos, [lines[idx].strip() for idx in predicted[0]])
    plt.xlabel('Confidence (%)')
    plt.title('Top 5 Predictions')

    plt.tight_layout()

    # Create results directory if it doesn't exist
    os.makedirs('results', exist_ok=True)

    # Save the plot
    output_path = os.path.join('results', os.path.basename(image_path).split('.')[0] + '_prediction.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


if __name__ == '__main__':
    image_path = 'data/beer_person.jpg'
    predict_image(image_path)
