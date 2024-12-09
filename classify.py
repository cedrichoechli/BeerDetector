import argparse
import numpy
import torch
import torch.nn as nn
import torchvision
from PIL import Image
from beer_names import BEER_CLASSES




class ResNet(nn.Module):
    """
    class for custom model, based on pretraind resnet50
    """
    def __init__(self, model_location='models/resnet50_beer_classification_40.pth'):
        super(ResNet, self).__init__()
        class_names = BEER_CLASSES

        # use pretrained resnet50
        self.resnet = torchvision.models.resnet50(pretrained=True)
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_ftrs, len(class_names))
        self.resnet.load_state_dict(torch.load(model_location, map_location=torch.device('cpu')))

        # isolate the feature blocks
        self.features = nn.Sequential(self.resnet.conv1,
                                      self.resnet.bn1,
                                      nn.ReLU(),
                                      nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False),
                                      self.resnet.layer1,
                                      self.resnet.layer2,
                                      self.resnet.layer3,
                                      self.resnet.layer4)

        # average pooling layer
        self.avgpool = self.resnet.avgpool

        # classifier
        self.classifier = self.resnet.fc

        # gradient placeholder
        self.gradient = None

    # hook for the gradients
    def activations_hook(self, grad):
        self.gradient = grad

    def get_gradient(self):
        return self.gradient

    def get_activations(self, x):
        return self.features(x)

    def forward(self, x):
        # extract the features
        x = self.features(x)

        # register the hook
        h = x.register_hook(self.activations_hook)

        # complete the forward pass
        x = self.avgpool(x)
        x = x.view((1, -1))
        x = self.classifier(x)

        return x




def beer_classification(img_location, model_location='models/resnet50_beer_classification_40.pth'):
    """
    classify beer using custom trained model
    """

    device = torch.device('cpu')
    # get classes
    class_names = BEER_CLASSES
    # init the resnet
    resnet = ResNet(model_location)
    # set the evaluation mode
    resnet.eval().to(device)

    #open image
    img = Image.open(img_location)

    test_transforms = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])])  # normalize images for R, G, B (both mean and SD)

    img = test_transforms(img)

    img = img.to(device)
    # add 1 dimension to tensor
    img = img.unsqueeze(0).type(torch.float32)
    # forward pass

    print(img.dtype)
    pred = resnet(img)

    # tranfors tensors with results to probabilities
    sm = torch.nn.Softmax(dim=1)
    probabilities = sm(pred)
    return probabilities, class_names[pred.argmax()]




if __name__ == '__main__':
    # construct argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', help='path to input image')
    args = vars(parser.parse_args())
    scores, top_guess = beer_classification(img_location=args['input'])

    # prints the beer brand and the according score
    for (score, beer) in zip(scores[0].tolist(), BEER_CLASSES):
        print(f'{beer}: {numpy.round(score, 3) * 100}%')
