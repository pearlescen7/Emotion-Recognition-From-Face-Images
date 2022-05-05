from torchvision import transforms
import torchvision.models as M
from torch.utils.data import random_split
from dataset import KDEFDataset, KDEFDataLoader
import torch.optim as optim
import torch.nn as nn
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, confusion_matrix
import numpy as np

kdef_dataset = KDEFDataset(transform=
transforms.Compose([
    transforms.Resize((224, 224)), 
    transforms.ToTensor()
    ]))

train_size = int(0.9*len(kdef_dataset))
test_size = (len(kdef_dataset) - train_size) // 2
val_size = len(kdef_dataset) - train_size - test_size
train_set, test_set, val_set = \
    random_split(kdef_dataset, [train_size, test_size, val_size], generator=torch.Generator().manual_seed(42))

batch_size = 32

train_loader = KDEFDataLoader(train_set, batch_size=batch_size)
test_loader = KDEFDataLoader(test_set, batch_size=batch_size)
val_loader = KDEFDataLoader(val_set, batch_size=batch_size)

print(f"{len(train_loader.dataset)} train data loaded.")
print(f"{len(test_loader.dataset)} test data loaded.")
print(f"{len(val_loader.dataset)} validation data loaded.")

EPOCHS = 100
LR = 0.001
MOMENTUM = 0.9
VALID_LOSS_PER_EPOCH = 4
TRAIN_LOSS_PER_EPOCH = 8
SAVE_PATH = "../models/VGG/vgg19_bn_KDEF.pt"

def generate_save_path(idx, save_path=SAVE_PATH):
    return save_path.removesuffix(".pt") + f"_{idx}.pt"

model = M.vgg19_bn(num_classes=kdef_dataset.num_classes).cuda()

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=LR, weight_decay=0.0005)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

micro_f1s = []
macro_f1s = []
confusion_mats = []
def test_loop():
    correct = 0
    total = 0

    categories = {"afraid","angry","disgusted","happy","neutral","sad","surprised"}
    category_correct = {category : 0 for category in categories}
    category_total = {category : 0 for category in categories}

    label2category = \
    {
        0 : "afraid"   ,
        1 : "angry"    , 
        2 : "disgusted", 
        3 : "happy"    , 
        4 : "neutral"  , 
        5 : "sad"      , 
        6 : "surprised"
    }

    micro_f1 = 0.0
    macro_f1 = 0.0
    confusion_mat = np.zeros((7, 7), dtype=np.int32)

    model.eval()
    with torch.no_grad():
        for data in test_loader:
            inputs = data["images"].cuda()
            labels = data["labels"]

            outputs = model(inputs).cpu()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            # print(labels, predicted)
            micro_f1 += f1_score(labels, predicted, average='micro') * len(data["labels"])
            macro_f1 += f1_score(labels, predicted, average='macro') * len(data["labels"])
            # if epoch == 0:
            confusion_mat += confusion_matrix(labels, predicted, labels=[0, 1, 2, 3, 4, 5, 6])
            
            for idx, (prediction, label) in enumerate(zip(predicted, labels)):
                if(prediction == label):
                    category_correct[label2category[label.item()]] += 1
                # else:
                    # print(f'Incorrect Guess Filename: {data["filenames"][idx]}, {prediction}')
                category_total[label2category[label.item()]] += 1

    print(f"Accuracy: {100*correct/total}%")
    micro_f1s.append(micro_f1 / len(test_loader.dataset))
    macro_f1s.append(macro_f1 / len(test_loader.dataset))
    confusion_mats.append(confusion_mat)
    for category in category_correct:
        print(f"Accuracy for {category}: {100*category_correct[category]/category_total[category]}%")

    model.train()
    

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

        if i % int(len(train_loader)/TRAIN_LOSS_PER_EPOCH) == 0 and i!=0:
            print(f"[EPOCH {epoch+1}, BATCH {i+1}, TOTAL {epoch*train_size + i*batch_size}] training loss: {running_loss/TRAIN_LOSS_PER_EPOCH:.8f}")
            train_losses.append((running_loss/TRAIN_LOSS_PER_EPOCH, epoch))
            running_loss = 0.0

        if i % int(len(train_loader)/VALID_LOSS_PER_EPOCH) == 0 and i!=0:
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
            validation_losses.append((valid_loss, epoch))
            print(f"[EPOCH {epoch+1}, BATCH {i+1}, TOTAL {epoch*train_size + i*batch_size}] validation loss: {valid_loss:.8f}")
            model.train()

    test_loop()

    if epoch%20==0 and epoch!=0:
        torch.save(model.state_dict(), generate_save_path(epoch))    
        
    scheduler.step()

torch.save(model.state_dict(), SAVE_PATH)

train_loss, train_epoch = tuple([list(tup) for tup in zip(*train_losses)])
valid_loss, valid_epoch = tuple([list(tup) for tup in zip(*validation_losses)])
plt.plot(train_epoch, train_loss)
plt.plot(valid_epoch, valid_loss)
plt.title('VGG19 (Batch Normalized) Loss Plot')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(["Train", "Validation"], loc='upper right')
plt.savefig("../models/losses.png")

plt.cla()
plt.plot(list(range(epoch + 1)), micro_f1s)
plt.plot(list(range(epoch + 1)), macro_f1s)
plt.title('F1 Scores')
plt.ylabel('F1 Score')
plt.xlabel('Epoch')
plt.legend(['Micro F1', 'Macro F1'], loc='upper right')
plt.savefig('../models/f1scores.png')
print(confusion_mats[-1])
