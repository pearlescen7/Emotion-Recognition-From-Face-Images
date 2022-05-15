import torch
import torchvision.models as M
from torchvision import transforms
from PIL import Image
import cv2
import torch
from dataset import KDEFDataset, KDEFDataLoader
from torch.utils.data import random_split
from efficientnet_pytorch import EfficientNet
from vgg_pytorch import VGG 


kdef_num_classes = 7

# model_path = "../models/EfficientNet/EfficientNet_b7_pretrained_KDEF.pt"
# model = EfficientNet.from_pretrained('efficientnet-b7')
# model._fc= torch.nn.Linear(in_features=model._fc.in_features, out_features=kdef_num_classes, bias=True)
# model.load_state_dict(torch.load(model_path))

# model_path = "../models/VGG/vgg19_bn_pretrained_KDEF.pt"
# model = VGG.from_pretrained('vgg19_bn', num_classes=kdef_num_classes)
# model.load_state_dict(torch.load(model_path))

model_path = "../models/VGG/vgg19_bn_KDEF.pt"
model = M.vgg19_bn(num_classes=kdef_num_classes)
model.load_state_dict(torch.load(model_path))

model.eval()
img_transforms = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.485, 0.456, 0.406],
        #                             std=[0.229, 0.224, 0.225])
        ])

idx2emote_and_color = \
{
    0 : ("afraid"   , (0, 0, 255)),
    1 : ("angry"    , (0, 215, 255)), 
    2 : ("disgusted", (133, 21, 199)), 
    3 : ("happy"    , (0, 255, 0)), 
    4 : ("neutral"  , (255, 255, 0)), 
    5 : ("sad"      , (255, 0, 0)), 
    6 : ("surprised", (255, 144, 30))
}

# kdef_dataset = KDEFDataset(transform=
# transforms.Compose([
#     transforms.Resize((224, 224)), 
#     transforms.ToTensor()
#     ]))

# train_size = int(0.7*len(kdef_dataset))
# test_size = (len(kdef_dataset) - train_size) // 2
# val_size = len(kdef_dataset) - train_size - test_size
# train_set, test_set, val_set = \
#     random_split(kdef_dataset, [train_size, test_size, val_size], generator=torch.Generator().manual_seed(42))

# batch_size = 32

# train_loader = KDEFDataLoader(train_set, batch_size=batch_size)
# test_loader = KDEFDataLoader(test_set, batch_size=batch_size)
# val_loader = KDEFDataLoader(val_set, batch_size=batch_size)

# for i, data in enumerate(test_loader):
#     print(data["filenames"])
#     if(i == 10):
#         break
# exit()

face_detector = cv2.CascadeClassifier("../data/haarcascades/haarcascade_frontalface_alt.xml")
fnames = [("happy-black-guy.jpg", "happy"), ("sad-man.jpg", "sad"), ("happy-furkan.jpg", "happy"), ("angry.png", "angry"), 
        ("happy-woman.jpg", "happy"), ("lena.png", "neutral"), ("surprised-man.jpg", "surprised")]

for fname in fnames:
    frame = cv2.imread(fname[0])
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(frame_gray, 1.1, 3)
    for (x,y,w,h) in faces:
        face = cv2.cvtColor(frame[y:y+h, x:x+w], cv2.COLOR_BGR2RGB)
        print(face.shape)
        pil_face = Image.fromarray(face)
        input = img_transforms(pil_face)
        # input.show()
        output = model(input.unsqueeze(0))
        data = torch.softmax(output.data, 1)
        print(f'filename: {fname[0]}')
        # print(f'tensor: {data}')
        _, predicted = torch.max(data, 1)
        emote, color = idx2emote_and_color[predicted.item()]
        print(f'predicted emotion: {emote}')
        print(f'true emotion: {fname[1]}')
        print(emote == fname[1])