import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
from model.mobilenet import MobileNet
from model.mobilenet_v2 import MobileNetV2
from model.resnet import ResNet18
from model.vgg import VGG

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=0)
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=16, shuffle=True, num_workers=0)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# net = MobileNet().cuda()
# net = MobileNetV2(n_class=10, input_size=32).cuda()
# net = ResNet18().cuda()
net = VGG("VGG19").cuda()

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)

criterion = criterion.cuda()

for epoch in range(120):
    net.train()
    for i, data in enumerate(trainloader, start=1):
        inputs, labels = data
        inputs = inputs.cuda()
        labels = labels.cuda()

        outputs = net(inputs)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss = loss.item()

        if i % 10 == 0:
            print('EPOCH %d loss: %f' % (epoch, running_loss))

    net.eval()

    correct = 0
    total = 0

    for test_data in testloader:
        test_images, test_labels = test_data
        test_images = test_images.cuda()
        test_labels = test_labels.cuda()

        test_outputs = net(test_images)
        predicted = torch.max(test_outputs, 1)[1].data.cpu().squeeze().numpy()

        total += float(test_labels.size(0))
        correct += float((predicted == test_labels.data.cpu().numpy()).astype(int).sum())

    print('Accuracy on the test images: %.2f %%' % (100 * correct / total))
