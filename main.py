import torch
from torchvision import transforms
from vgg19 import vgg19
from PIL import Image
import os
from os import listdir
import io

def image_loader(image_name):
    image = Image.open(image_name)
    # fake batch dimension required to fit network's input dimensions
    image = loader(image).unsqueeze(0)
    return image.to(device)


# image_path = "/content/Gaurav.jpg"
feature_layer = "pool5"

folder_path = "C:\Users\ASUS\Desktop\Richa_GAN\"
for images in os.listdir(folder_path):
  if(images.endswith(".jpg")):
    image_path = folder_path + images
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    normalization_mean = [0.485, 0.456, 0.406]
    normalization_std = [0.229, 0.224, 0.225]

    loader  = transforms.Compose([transforms.ToTensor(), 
                                    transforms.Normalize(mean = normalization_mean, std = normalization_std)])

    vgg = vgg19().to(device)
    img = image_loader(image_path)

    vgg_features = vgg(img)
    feature = getattr(vgg_features, feature_layer)
    # print(feature)
    size = len(images)
    tensor_file = images[:size-4]+"_tensor.pt"
    torch.save(feature,tensor_file)
    buffer = io.BytesIO()
    torch.save(feature,buffer)