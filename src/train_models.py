from torchvision import transforms
import torchvision.models as M
from torch.utils.data import random_split
from dataset import KDEFDataset, KDEFDataLoader
import torch.optim as optim
import torch.nn as nn
import torch
import matplotlib.pyplot as plt

kdef_dataset = KDEFDataset(transform=
transforms.Compose([
    transforms.Resize(256), 
    transforms.CenterCrop(224), 
    transforms.ToTensor()
    ]))

train_size = int(0.7*len(kdef_dataset))
test_size = (len(kdef_dataset) - train_size) // 2
val_size = len(kdef_dataset) - train_size - test_size
train_set, test_set, val_set = random_split(kdef_dataset, [train_size, test_size, val_size])

batch_size = 4

train_loader = KDEFDataLoader(train_set, batch_size=batch_size)
test_loader = KDEFDataLoader(test_set, batch_size=batch_size)
val_loader = KDEFDataLoader(val_set, batch_size=batch_size)

print(f"{len(train_loader.dataset)} train data loaded.")
print(f"{len(test_loader.dataset)} test data loaded.")
print(f"{len(val_loader.dataset)} validation data loaded.")

EPOCHS = 80
LR = 0.001
MOMENTUM = 0.9
VALID_PER_EPOCH = 4
SAVE_PATH = "../models/VGG/vgg11_bn_KDEF.pt"

def generate_save_path(idx, save_path=SAVE_PATH):
    return save_path.removesuffix(".pt") + f"_{idx}.pt"

model = M.vgg11_bn(num_classes=kdef_dataset.num_classes).cuda()

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=LR, momentum=MOMENTUM, weight_decay=0.0005)

train_losses = []
validation_losses = []
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

        if i % 25 == 24:
            print(f"[EPOCH {epoch+1}, BATCH {i+1}, TOTAL {epoch*train_size + i*batch_size}] training loss: {running_loss/25:.8f}")
            train_losses.append(running_loss)
            running_loss = 0.0

        if i % int(len(train_loader.dataset)/VALID_PER_EPOCH) == 0 and i!=0:
            model.eval()
            valid_loss = 0.0
            with torch.no_grad():
                for j, data in enumerate(val_loader):
                    val_inputs = data["images"].cuda()
                    val_labels = data["labels"].cuda()

                    output = model(inputs)
                    loss = criterion(output, labels)
            
                    valid_loss += loss.item()
            valid_loss /= len(val_loader)
            validation_losses.append(valid_loss)
            print(f"[EPOCH {epoch+1}, BATCH {i+1}, TOTAL {epoch*train_size + i*batch_size}] validation loss: {valid_loss:.8f}")
            model.train()

    if epoch%20==0 and epoch!=0:
        torch.save(model.state_dict(), generate_save_path(epoch))

torch.save(model.state_dict(), SAVE_PATH)

plt.plot(train_losses)
plt.plot(validation_losses)
plt.title('VGG11 (Batch Normalized) Loss Plot')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(["Train", "Validation"], loc='upper right')
plt.savefig("../models/losses.png")

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