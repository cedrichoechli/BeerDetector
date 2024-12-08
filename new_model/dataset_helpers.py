import os
import random
import detect
from beer_names import BEER_CLASSES



def split_trainval(folder_beers='data/original', fraction_train = 0.7):
    """
    Splits the images in folder_beers into training and validation dataset with a default ratio of 70:30
    """
    brands = os.listdir(folder_beers)

    #create train and val folders
    os.makedirs(folder_beers+'/train')
    os.makedirs(folder_beers+'/val')

    for brand in brands:
        # get images
        images = os.listdir(folder_beers + '/' + brand)

        # select random images to train / validate
        n_train = int(round(len(images)*fraction_train, 0))
        images_train = random.sample(images, n_train)
        images_val = [x for x in images if x not in images_train]

        # move images to new folders
        os.makedirs(folder_beers + '/train' + '/' + brand)
        for image in images_train:
            os.rename(src=folder_beers + '/' + brand + '/' + image, dst=folder_beers + '/train' + '/' + brand + '/' + image)

        os.makedirs(folder_beers + '/val' + '/' + brand)
        for image in images_val:
            os.rename(src=folder_beers + '/' + brand + '/' + image, dst=folder_beers + '/val' + '/' + brand + '/' + image)

        os.rmdir(folder_beers + '/' + brand)



def crop_images_to_directory():
    """
    Takes raw pictures in data/original/ and crops them the data/detected/ using beer detection model
    """
    for brand in os.listdir('data/original/train/'):
        # create folders in data/detected
        os.makedirs(f'data/detected/train/{brand}', exist_ok=True)
        os.makedirs(f'data/detected/val/{brand}', exist_ok=True)

        # crop validation images to data/detected folder
        for image in os.listdir(f'data/original/val/{brand}'):
            try:
                print(f'Running detection on {image}')
                detect.detect(f'data/original/val/{brand}/{image}', save_cropped_image=True, output_filename=f'data/detected/val/{brand}/{image}')
            except:
                print(f"Error at {image}")


        # crop training images to data/detected folder
        for image in os.listdir(f'data/original/train/{brand}'):
            try:
                print(f'Running detection on {image}')
                detect.detect(f'data/original/train/{brand}/{image}', save_cropped_image=True, output_filename=f'data/detected/train/{brand}/{image}')
            except:
                print(f"Error at {image}")




if __name__ == '__main__':
    #split_trainval()
    #crop_images_to_directory()
    pass
