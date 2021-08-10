import torch
import torchvision
import PIL
import os
import collections
torch.normal
import types
import scipy.spatial
import numpy

import argparse

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("-g", "--gallery-variation", default="normal",choices = ("normal", "happy", "leftlight", "rightlight", "glasses", "sad","seeley", "wink", "centerlight", "surprised", "noglasses"), help = "Select variation to put into the gallery")
parser.add_argument("-n", "--network-depth", type=int, choices = (18,34,50,101,152), default = 50, help = "select the depth of the residual network")
parser.add_argument("-p", "--pretrained", action="store_false",help = "Disable the pretrained netword")
parser.add_argument("-x", "--device", choices=("cuda","cpu"), default = "cpu",help = "Select the device to run on")
parser.add_argument("-d", "--distance-function", choices = ("euclidean", "cosine"), default = "cosine", help = "Select the distance function for computing distance values")
parser.add_argument("-r", "--raw-pixel-values", action = "store_true", help = "Use raw pixel values only")

args = parser.parse_args()

# Re-implent forward function to extract the deep features
# This function is a copy from the original implentation, see
# https://github.com/pytorch/vision/blob/37eb37a836fbc2c26197dfaf76d2a3f4f39f15df/torchvision/models/resnet.py#L230

def _forward_impl(self, x):
  x = self.conv1(x)
  x = self.bn1(x)
  x = self.relu(x)
  x = self.maxpool(x)

  x = self.layer1(x)
  x = self.layer2(x)
  x = self.layer3(x)
  x = self.layer4(x)

  x = self.avgpool(x)
  x = torch.flatten(x, 1)
  # Do not apply the last fully-connect layer
  # x = self.fc(x)

  # Return deep features
  return x

# set device and distance function
device = torch.device(args.device)

distance_function = {
    "cosine" : scipy.spatial.distance.cosine,
    "euclidean" : scipy.spatial.distance.euclidean,
}[args.distance_function]

# load (pretrain) network
network = {
    18:torchvision.models.resnet18,
    34:torchvision.models.resnet34,
    50:torchvision.models.resnet50,
    101:torchvision.models.resnet101,
    152:torchvision.models.resnet152,
}[args.network_depth](pretrained=args.pretrained)

network.eval()
network.to(device)

# overwrite its forward function
network._forward_impl = types.MethodType(_forward_impl,network)

# define a set of image transforms
# 主要学习torchvision.transforms，
# 包括常用的图像操作，例如：随机切割，旋转，数据类型转换，
# 图像到tensor ,numpy 数组到tensor , tensor 到 图像等。
# transforms.Compose可以组合几个变换连续一起操作
transform = torchvision.transforms.Compose([
  # scale smaller size to 300 pixels
  torchvision.transforms.Resize(300),
  # take the certer-crop from the image
  torchvision.transforms.CenterCrop(224),
  # transform this PIL image to a tensor
  torchvision.transforms.ToTensor(),
  # convert 1-plane gray-scale image to 3 plane
  lambda x : x.repeat(3,1,1),
  # subtract mean and divide by standard deviation
  torchvision.transforms.Normalize(
      mean = [0.485, 0.456, 0.406],
      std = [0.229, 0.224, 0.225]
  ),
  # add the batch dimension (requred by pytorch)
  lambda x : x.unsqueeze(0)        # 加x1 = 1                                  
])

# extract the deep features from the image
def extract(path):
  with torch.no_grad():
    # load image using PIL
    image = PIL.Image.open(path)
    # turn image into tensor
    tensor = transform(image)
    # extract deep feature from network
    feature = network(tensor.to(device))
    # return the fratures as flatted numpy vector
    q  = feature.cpu().numpy.flatten()
    return q
# optional
# variation to only extract pixel values from the image
def extrat_raw(path):
  # load image using PIL 
  image = PIL.Image.open(path)
  # turn image into tensor
  tensor = transform(image)
  # convert to numpy and turn into 1D array
  return tensor.numpy().flatten()

if args.raw_pixel_values:
  extract = extrat_raw

# load gallery
print("Enrolling Gallery")
gallery = {}
for index in range(1,16): # 15个人
  # get subject information
  subject = F"subject{index:02d}" 
  # store deep feature for object
  gallery[subject] = extract(F"yalefaces/{subject}.{args.gallery_variation}")

# perform probing
print("Scoring")
# store number of correct identifications and total number of identifacations per variation

# 为字典的没有的key提供一个默认的值。参数应该是一个函数，当没有参数调用时返回默认值。如果没有传递任何内容，则默认为None。
variations = collections.defaultdict(lambda:[0,0])
# iterate over all files in the directory
for filename in os.listdir("yalefaces"):
  # split off the subject and the filname extension
  probe_subject, variation = os.path.splitext(filename)
  # ignore some non-appropriate files
  if variation in (".text",".gif"):
    continue

  # compute the path and ignore non-files
  path = os.path.join("/Users/xinyi/Desktop/2021Spring/Deep Learning/code/yalefaces", filename)
  if not os.path.isfile(path):
    continue

  # extract deep feature
  probe_feature = extract(path)
  # find the most similar face in the gallery
  min_distance = 1e8
  best_subject = None
  for gallery_subject, gallery_feature in gallery.items():
    # comlute distance between gallery and prob feature
    distance = distance_function(gallery_feature, probe_feature)
    if distance < min_distance:
      # store the subject with the lowest distance
      min_distance = distance
      best_subject = gallery_subject

  # check if face was recognized correctly
  variations[variation][0] += best_subject == probe_subject
  variations[variation][1] += 1

  

# print recognition rates for each variation
for variation, results in variations.items():
  print (F"Recognized {results[0]:2} of {results[1]} faces ({100*results[1]:6.2f}%) with {variation[1:]}")

# compute the total and print
total = numpy.sum(list(variations.values()),axis=0)
print (F"\nTotal: {total[0]:3} of {total[1]:3} faces ({100*total[1]:6.2f}%)")
