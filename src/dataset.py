from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader
from torchvision import transforms
import torch
from PIL import Image
import os, copy
import matplotlib.pyplot as plt
import cv2

class LabeledImage():
    def __init__(self, filename, image, label):
        self.filename = filename
        self.image = image
        self.label = label

class KDEFDataset(Dataset):
    def __init__(self, root_dir="..\\data\\KDEF_Filtered\\", transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.num_classes = 7
        self.data = {}

        self.emote2label = \
        {
            "afraid"    : 0,
            "angry"     : 1, 
            "disgusted" : 2, 
            "happy"     : 3, 
            "neutral"   : 4, 
            "sad"       : 5, 
            "surprised" : 6
        }

        self.label2emote = \
        {
            0 : "afraid"   ,
            1 : "angry"    , 
            2 : "disgusted", 
            3 : "happy"    , 
            4 : "neutral"  , 
            5 : "sad"      , 
            6 : "surprised"
        }

        face_detector = cv2.CascadeClassifier("../data/haarcascades/haarcascade_frontalface_alt.xml")

        #Dataset is small enough to load all of it to memory
        data_idx = 0
        for dirname, _, filenames in os.walk(root_dir):
            label = dirname.split("\\")[-1]
            for filename in filenames:
                img_path = os.path.join(dirname,filename)

                # The output image might be different depending on its type: when downsampling,
                # the interpolation of PIL images and tensors is slightly different,
                # because PIL applies antialiasing. This may lead to significant differences in the performance of a network.
                # Therefore, it is preferable to train and serve a model with the same input types. 
                # See also below the antialias parameter, which can help making the output of PIL images and tensors closer.
                img = cv2.imread(img_path)
                img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                faces = face_detector.detectMultiScale(img_gray, 1.1, 4)
                for (x,y,w,h) in faces:
                    face = cv2.cvtColor(img[y:y+h, x:x+w], cv2.COLOR_BGR2RGB)
                    pil_face = Image.fromarray(face)

                self.data[data_idx] = LabeledImage(filename, pil_face, self.emote2label[label])
                data_idx += 1

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if idx == len(self):
            raise IndexError
        
        sample = copy.deepcopy(self.data[idx])

        if self.transform:
            sample.image = self.transform(sample.image)

        return sample

    def collator(obj_list):
        images = []
        labels = []
        filenames = []
        for obj in obj_list:
            images.append(obj.image)
            labels.append(obj.label)
            filenames.append(obj.filename)
        return {"images" : torch.stack(images),
                "labels" : torch.tensor(labels, dtype=torch.long),
                "filenames" : filenames}

class KDEFDataLoader(DataLoader):
    def __init__(self, dataset, batch_size=4, shuffle=True):
        super().__init__(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0, collate_fn=KDEFDataset.collator)

def main():

    # kdef_dataset = KDEFDataset(
    #     transform=transforms.Compose([
    #         transforms.Resize(256),
    #         transforms.CenterCrop(224),
    #         transforms.ToTensor()
    #     ]))

    # print(len(kdef_dataset))
    # # for idx, data in enumerate(kdef_dataset):
    # #     print(idx, data.filename, data.label)
    # #     image = data.image.permute((1, 2, 0)).contiguous()
    # #     plt.imshow(image)
    # #     plt.show(block=True)
    # #     break

    # data_loader = DataLoader(kdef_dataset, batch_size=4, shuffle=True, num_workers=0, collate_fn=KDEFDataset.collator)

    kdef_dataset = KDEFDataset(
        transform=transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor()
        ]))

    data_loader = KDEFDataLoader(kdef_dataset)

    for i, batch in enumerate(data_loader):
        print(i, batch["images"].size())
        break


if __name__ == "__main__":
    main()