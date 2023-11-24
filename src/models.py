import torch
import torch.nn as nn
import torchvision.models
import torch.nn.functional as F



def get_pretrained_VGG16():
    # Загружаем предварительно обученную модель VGG16
    vgg16 = torchvision.models.vgg16(pretrained=True)
    # нужное число выходных нейронов
    OUTPUT_NEURONS = 5

    # У вгг последние слои находятся в блоке classifier и он состоит из 6 слоев, мы обращаемся
    # к этому блоку, к последнему (6) слою и забираем in_features, так как он понадобится для изменения
    # последнего слоя
    num_of_in_features = vgg16.classifier[6].in_features

    # заменяем число выходных нейронов своим числом
    vgg16.classifier[6] = torch.nn.Linear(num_of_in_features, OUTPUT_NEURONS)

    return vgg16


def get_efficient_net(size='m', pretrained=True):
    print('Входное должно иметь размер 224x224')

    models = {'s': torchvision.models.efficientnet_v2_s(pretrained=pretrained),
              'm': torchvision.models.efficientnet_v2_m(pretrained=pretrained),
              'l': torchvision.models.efficientnet_v2_l(pretrained=pretrained)}

    effnet = models[size]

    OUTPUT_NEURONS = 5

    num_of_in_features = effnet.classifier[-1].in_features

    effnet.classifier[-1] = torch.nn.Linear(num_of_in_features, OUTPUT_NEURONS)

    return effnet




class VGG16(nn.Module):
    def __init__(self, img_size):
        super().__init__()

        self.img_size = img_size

        self.conv_layers = nn.ModuleList([])

        layers_with_maxpool = [2, 4, 7, 10, 13]
        kernel_count = {1: 64, 2: 64, 3: 128, 4: 128, 5: 256, 6: 256,
                        7: 256, 8: 512, 9: 512, 10: 512, 11: 512, 12: 512, 13: 512}

        for layer in range(1, 14):
            cur_kernel_count = kernel_count[layer]
            self.conv_layers.append(nn.Conv2d(cur_kernel_count, cur_kernel_count, kernel_size=3, padding=1))
            self.conv_layers.append(nn.BatchNorm2d(cur_kernel_count))
            self.conv_layers.append(nn.ReLU(inplace=True))

            if layer in layers_with_maxpool:
                self.conv_layers.append(nn.MaxPool2d(2, 2))

        self.fc_layers = nn.ModuleList([])
        neuron_count = {14: (self.img_size/2**5)*512, 15: 4096}

        for layer in range(14, 16):
            cur_neuron_count = neuron_count[layer]
            self.fc_layers.append(nn.Dropout(0.5))
            self.fc_layers.append(nn.Linear(int(cur_neuron_count), 4096))
            self.fc_layers.append(nn.ReLU(inplace=True))

        self.fc_layers.append(nn.Linear(4096, 5))  # 5 - P, x, y, w, h

    def forward(self, out):

        for cur_layer in self.conv_layers:
            out = cur_layer(out)

        out = torch.flatten(out, 1)

        for cur_layer in self.fc_layers:
            out = cur_layer(out)

        return out


class VGG4(nn.Module):
    def __init__(self, num_classes, img_size):
        super(VGG4, self).__init__()

        # Convolutional layers for feature extraction
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Additional convolutional layer
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.relu4 = nn.ReLU()
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Fully connected layers for classification
        self.fc1 = nn.Linear(128 * (img_size // 16) * (img_size // 16), 128)
        self.relu5 = nn.ReLU()
        self.fc2 = nn.Linear(128, 64)
        self.relu6 = nn.ReLU()
        self.fc3 = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)

        x = self.conv3(x)
        x = self.relu3(x)
        x = self.pool3(x)

        # Additional convolutional layer
        x = self.conv4(x)
        x = self.relu4(x)
        x = self.pool4(x)

        # Flatten the data before passing it to fully connected layers
        x = x.view(x.size(0), -1)

        x = self.fc1(x)
        x = self.relu5(x)
        x = self.fc2(x)
        x = self.relu6(x)
        x = self.fc3(x)

        return x



class InspectorGadjet(nn.Module):
    def __init__(self):
        super(InspectorGadjet, self).__init__()
        
        # Основные сверточные слои
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )
        
        # Для классификации (лицо или фон)
        # self.classifier = nn.Sequential(
        #     nn.Linear(512 * 4 * 4, 1024),  # Предположим, что размер входного изображения 96x96
        #     nn.ReLU(inplace=True),
        #     nn.Dropout(0.05),
        #     nn.Linear(1024, 512),
        #     nn.Sigmoid(),
        #     nn.Dropout(0.05),
        #     nn.Linear(512, 1)
        # )

        # Для локализации 
        self.regressor = nn.Sequential(
            nn.Linear(512 * 4 * 4, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.05),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.05),
            nn.Linear(512, 5)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.shape[0], -1)  # Преобразуем в 1D тензор
        # classification_output = self.classifier(x)
        regression_output = self.regressor(x)
        return regression_output


class ConvEmbedding(nn.Module):
    def __init__(self, pic_size=160, emb_size=512):
        super().__init__()

        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(512),
            nn.MaxPool2d(2, 2)
        )

        dims = int(pic_size / 32)
        self.emb_size = emb_size

        self.fc1 = nn.Linear(dims * dims * 512, self.emb_size)
        self.fc2 = nn.Linear(self.emb_size, self.emb_size)

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.fc2(x)
        return x

class Triplet(nn.Module):
    
    def __init__(self, encoder):
        super(Triplet, self).__init__()
        
        self.encoder = encoder
        
    def forward(self, anchor, pos, neg):
        anchor_embedding = self.encoder(anchor)
        pos_embedding = self.encoder(pos)
        neg_embedding = self.encoder(neg)
        
        return anchor_embedding, pos_embedding, neg_embedding


def combined_loss(pred_class, pred_bbox, target):
    # Разделяем целевой тензор на класс и ограничивающую рамку
    target_class = target[:, 0].float()  # Shape: [batch_size, 1]
    target_bbox = target[:, 1:]  # Shape: [batch_size, 4]

    # Compute the classification loss
    loss_class = F.mse_loss(pred_class.squeeze(), target_class)

    # Compute the regression loss
    loss_bbox = F.smooth_l1_loss(pred_bbox, target_bbox)
    
    # Here, you can assign weights if you want to give different importance to the losses
    combined_loss = loss_class + loss_bbox

    return combined_loss


class InceptionModule(nn.Module):
    def __init__(self, in_channels, n1x1, n3x3red, n3x3, n5x5red, n5x5, pool_proj):
        super(InceptionModule, self).__init__()
        # 1x1 conv branch
        self.b1 = nn.Sequential(
            nn.Conv2d(in_channels, n1x1, kernel_size=1),
            nn.BatchNorm2d(n1x1),
            nn.ReLU(inplace=True)
        )

        # 1x1 conv -> 3x3 conv branch
        self.b2 = nn.Sequential(
            nn.Conv2d(in_channels, n3x3red, kernel_size=1),
            nn.BatchNorm2d(n3x3red),
            nn.ReLU(inplace=True),
            nn.Conv2d(n3x3red, n3x3, kernel_size=3, padding=1),
            nn.BatchNorm2d(n3x3),
            nn.ReLU(inplace=True)
        )

        # 1x1 conv -> 5x5 conv branch
        self.b3 = nn.Sequential(
            nn.Conv2d(in_channels, n5x5red, kernel_size=1),
            nn.BatchNorm2d(n5x5red),
            nn.ReLU(inplace=True),
            nn.Conv2d(n5x5red, n5x5, kernel_size=5, padding=2),
            nn.BatchNorm2d(n5x5),
            nn.ReLU(inplace=True)
        )

        # 3x3 pool -> 1x1 conv branch
        self.b4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels, pool_proj, kernel_size=1),
            nn.BatchNorm2d(pool_proj),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return torch.cat([self.b1(x), self.b2(x), self.b3(x), self.b4(x)], 1)

class GoogLeNet(nn.Module):
    def __init__(self, num_classes=1000):
        super(GoogLeNet, self).__init__()
        self.pre_layers = nn.Sequential(
            nn.Conv2d(3, 192, kernel_size=3, padding=1),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True)
        )

        self.a3 = InceptionModule(192, 64, 96, 128, 16, 32, 32)
        self.b3 = InceptionModule(256, 128, 128, 192, 32, 96, 64)

        # max pool

        self.a4 = InceptionModule(480, 192, 96, 208, 16, 48, 64)
        self.b4 = InceptionModule(512, 160, 112, 224, 24, 64, 64)
        self.c4 = InceptionModule(512, 128, 128, 256, 24, 64, 64)
        self.d4 = InceptionModule(512, 112, 144, 288, 32, 64, 64)
        self.e4 = InceptionModule(528, 256, 160, 320, 32, 128, 128)

        # max pool

        self.a5 = InceptionModule(832, 256, 160, 320, 32, 128, 128)
        self.b5 = InceptionModule(832, 384, 192, 384, 48, 128, 128)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.4)
        self.fc = nn.Linear(1024, num_classes)

    def forward(self, x):
        x = self.pre_layers(x)
        x = self.a3(x)
        x = self.b3(x)
        x = F.max_pool2d(x, kernel_size=3, stride=2, padding=1)
        x = self.a4(x)
        x = self.b4(x)
        x = self.c4(x)
        x = self.d4(x)
        x = self.e4(x)
        x = F.max_pool2d(x, kernel_size=3, stride=2, padding=1)
        x = self.a5(x)
        x = self.b5(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)  # Flatten the output
        x = self.dropout(x)
        x = self.fc(x)
        return x
