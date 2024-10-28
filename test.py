import torch
import torchvision.models as models
from torchvision import transforms
from PIL import Image



model = models.resnet50(pretrained=True)


def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    image = Image.open(image_path)
    image = transform(image)
    image = image.unsqueeze(0)

    return image


def predict_image(image_path):
    model.eval()

    input_image = preprocess_image(image_path)

    with torch.no_grad():
        output = model(input_image)

    # get topn elements from output tensor
    topn = 5
    values, predicted = torch.topk(output, topn)
  
    # get classname from ID
    with open('classes.txt') as file:
        lines = file.readlines()

    # print predicted items and their corresponding class name
    for i in predicted[0]:
        print(f'{i} - {lines[i]}')



if __name__ == '__main__':
    image_path = 'data/beer_bottle.jpg'
    predict_image(image_path)
