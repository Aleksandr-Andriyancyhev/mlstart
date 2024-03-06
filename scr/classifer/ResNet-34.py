import torch
import torchvision.models as models
import torchvision.transforms as transforms
import torchvision.datasets as dataset
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim

device = None
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

# определим функцию, которая будет вычислять точность модели на итерации
def calculate_accuracy(y_pred, y):
    # находим количество верных совпадений лейбла и выходного класса по каждому примеру в батче
    top_pred = y_pred.argmax(1, keepdim = True)
    correct = top_pred.eq(y.view_as(top_pred)).sum()
    # посчитаем точность, которая равна отношению количества верных совпадений к общему числу примеров в батче
    acc = correct.float() / y.shape[0]
    return acc

# функция, отвечающая за обучение сети на одной эпохе
def train(model, dataloader, optimizer, loss_function, device):
    # определим значения точности и потерь на старте эпохи
    epoch_acc = 0
    epoch_loss = 0
    # переведем модель в режим тренировки
    model.train()
    # для каждого батча в даталоадере
    for (images, labels) in dataloader:

        # отправляем изображения и метки на устройство
        images = images.to(device)
        labels = labels.to(device)
        # обнуляем градиенты
        optimizer.zero_grad()
        # вычислим выходы сети на данном батче
        predicts = model(images)
        # вычислим величину потерь на данном батче
        loss    = loss_function(predicts, labels)
        # вычислим точность на данном батче
        acc = calculate_accuracy(predicts, labels)
        # вычислим значения градиентов на батче
        loss.backward()
        # корректируем веса
        optimizer.step()
        # прибавим значения потерь и точности на батче 
        epoch_loss += loss.item()
        epoch_acc  += acc.item()
    # возвращаем величину потерь и точность на эпохе
    return epoch_loss / len(dataloader),  epoch_acc / len(dataloader)

# функция, отвечающая за проверку модели на одной эпохе
def evaluate(model, dataloader, loss_function, device):
    
    # определим начальные величины потерь и точности
    epoch_acc = 0
    epoch_loss = 0
    # переведем модель в режим валидации
    model.eval()
    # указываем, что градиенты вычислять не нужно
    with torch.no_grad():
      
        # для каждого батча в даталоадере
        for images, labels in dataloader:
            # переносим изображения и лейблы на устройство
            images = images.to(device)
            labels = labels.to(device)
            # вычислим выходы сети на батче
            predicts = model(images)
            # вычислим величину потерь на батче
            loss = loss_function(predicts, labels)
            # вычислим точность на батче
            acc  = calculate_accuracy(predicts, labels)
            # прибавим значения потерь и точности на батче к общему
            epoch_loss += loss.item()
            epoch_acc  += acc.item()
    # возвращаем величину потерь и точность на эпохе
    return epoch_loss / len(dataloader),  epoch_acc / len(dataloader)

       
 

transform = transforms.Compose([
      transforms.Resize(256),
      transforms.CenterCrop(224),
      transforms.ToTensor(),
      transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

train_path = r"C:\\Users\\profi\\Desktop\\ml\\ships_vs_planes\\train"
test_path  = r"C:\\Users\\profi\\Desktop\\ml\\ships_vs_planes\\test"

train_data = dataset.ImageFolder(train_path, transform)
test_data = dataset.ImageFolder(test_path, transform)
print(type(train_data))
print(type(test_data))

print(train_data.classes)
print(test_data.classes)

train_loader_1 = DataLoader(train_data, batch_size=16, shuffle=True)
test_loader_1  = DataLoader(train_data, batch_size=16, shuffle=True)

pretrained_resnet34 = models.resnet34(pretrained=True)

for name, param in pretrained_resnet34.named_parameters():
    param.requires_grad = False

pretrained_resnet34.fc = nn.Sequential(
    nn.Linear(pretrained_resnet34.fc.in_features, 500),
    nn.ReLU(),
    nn.Dropout(),
    nn.Linear(500, 2)
)
pretrained_resnet34.to(device)

# попробуем обучить!

epochs = 5
optimizer = optim.Adam(pretrained_resnet34.parameters(), lr=0.001)
loss_function = nn.CrossEntropyLoss()
best_loss = 1000000
best_acc = 0
for epoch in range(epochs):
    
    train_loss, train_acc = train(pretrained_resnet34, train_loader_1, optimizer, loss_function, device)

    test_loss, test_acc   = evaluate(pretrained_resnet34, test_loader_1, loss_function, device)

    print(f'Epoch: {epoch+1}')
    print(f'\tTrain Loss: {train_loss} | Train Acc: {train_acc*100}%')
    print(f'\tTest. Loss: {test_loss} |   Test Acc: {test_acc*100}%')


    if test_loss < best_loss:
        best_loss = test_loss
        torch.save(pretrained_resnet34, "resnet34_best_loss.pth")
