import torch
import torchvision
import PIL
import os
import csv

from stargan.model import Generator

# select an image
input_image = "stargan/data/celeba/images/000053.jpg"
# input_image = "Ich.jpg"

# The list of attributes that StarGAN uses
attributes = ["Black_Hair", "Blond_Hair", "Brown_Hair", "Male", "Young"]

# load the original labels for the input_image
def load_original_labels():
    with open("stargan/data/celeba/list_attr_celeba.txt") as f:
        r = csv.reader(f, delimiter=" ", skipinitialspace=True)
        # skip first line
        next(r)
        # read header
        header = next(r)
        base = os.path.basename(input_image)
        # parse file
        for line in r:
            if line[0] == base:
                # convert {-1,1} labels into {0,1} labels
                return torch.tensor([(float(line[header.index(a)+1]) + 1) / 2 for a in attributes])

# load labels
tt = load_original_labels()
# labels not found -> image of myself
if tt is None:
    tt = torch.tensor([0,0,1,1,1])
# add batch index
orig_target = tt.unsqueeze(0)
print(orig_target)

# load image via PIL
image = PIL.Image.open(input_image)

# define transform according to stargan implementation
transform = torchvision.transforms.Compose([
    torchvision.transforms.CenterCrop(image.size[0]),
    torchvision.transforms.Resize(128),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=(0.5,0.5,0.5), std=(0.5,0.5,0.5))                                 
])

# load generator network
G_path = "stargan/stargan/celeba_128/models/200000-G.ckpt"
G = Generator(64, len(attributes),6)
G.load_state_dict(torch.load(G_path, map_location=lambda storage, loc: storage))

# save function image
topil = torchvision.transforms.ToPILImage()

def save_image(tensor, filename):
    # image was normalized by subtracting mean 0.5 and dividing by 0.5
    # Undo this first, and clamp to valid pixel range
    tensor = torch.clamp(tensor * 0.5 / 0.5, 0, 1)
    # convert to PIL image
    image = topil(tensor)
    # and save the image
    image.save(filename)

# transform image to tensor
tensor = transform(image)

# write original image
base = os.path.splitext(os.path.basename(input_image))[0]
filename = os.path.join("image", f"{base}.png")
save_image(tensor, filename)

# modify several attributes
for i in range(len(attributes)):
    # reset all attributes to 0
    target = torch.zeros(orig_target.shape)
    for j in (3,4):
        # copy gender and age from original
        target[0,j] = orig_target[0,j]
    # change specific attribute
    target[0,i] = 1-orig_target[0,i]        
    print(target)

    # generate image from original image and modified target attribute vector
    generated = G(tensor.unsqueeze(0), target).squeeze(0)

    # save generated image
    filename = os.path.join("images", f"{base}-{attributes[i]}-{int(target[0,i])}.png")
    save_image(generated, filename)