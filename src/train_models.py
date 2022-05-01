from torchvision import transforms
import torchvision.models as M
from torch.utils.data import random_split
from dataset import KDEFDataset, KDEFDataLoader
import torch.optim as optim
import torch.nn as nn
import torch

kdef_dataset = KDEFDataset(transform=
transforms.Compose([
    transforms.Resize(256), 
    transforms.CenterCrop(224), 
    transforms.ToTensor()
    ]))

train_size = int(0.8*len(kdef_dataset))
test_size = len(kdef_dataset) - train_size
train_set, test_set = random_split(kdef_dataset, [train_size, test_size])

batch_size = 32

train_loader = KDEFDataLoader(train_set, batch_size=batch_size)
test_loader = KDEFDataLoader(test_set, batch_size=batch_size)

print(f"{len(train_loader)*batch_size} train data loaded.")
print(f"{len(test_loader)*batch_size} test data loaded.")

EPOCHS = 100
LR = 0.001
MOMENTUM = 0.9
SAVE_PATH = "../models/VGG/vgg11_bn_KDEF.pt"

model = M.vgg11_bn(num_classes=kdef_dataset.num_classes).cuda()

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=LR, momentum=MOMENTUM, weight_decay=0.0005)

model.train()
for epoch in range(EPOCHS):
    running_loss = 0.0
    for i, data in enumerate(train_loader):
        inputs = data["images"].cuda()
        labels = data["labels"].cuda()

        optimizer.zero_grad()
        output = model(inputs)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        if i % 10 == 9:
            print(f"[EPOCH {epoch+1}, BATCH {i+1}, TOTAL {epoch*train_size + i*batch_size}] loss: {running_loss/100:.8f}")
            running_loss = 0.0

torch.save(model.state_dict(), SAVE_PATH)

correct = 0
total = 0

model.eval()
with torch.no_grad():
    for data in test_loader:
        inputs = data["images"].cuda()
        labels = data["labels"]

        outputs = model(inputs).cpu()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"Accuracy: {100*correct/total}%")