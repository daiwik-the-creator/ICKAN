import os
import time

from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader

from torch.utils.data import DataLoader
from torchvision import transforms

import os
import torch.optim as optim
import torch.nn as nn
import json
import torch
from torch.utils.data import Dataset
import librosa
import numpy as np


def save_checkpoint(model, optimizer, epoch, loss, checkpoint_dir):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }
    checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch}.pth')
    torch.save(checkpoint, checkpoint_path)
    print(f"Checkpoint saved: {checkpoint_path}")


def load_checkpoint(model, optimizer, checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    print(f"Loaded checkpoint from epoch {epoch} with loss {loss}")
    return epoch, loss


class AudioDataset_MFCC(Dataset):
    def __init__(self, file_paths, labels, sample_rate=22050, n_mels=64, n_fft=1024, hop_length=512, max_length=431,
                 transform=None):
        self.file_paths = file_paths
        self.labels = labels
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.n_mfcc = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.max_length = max_length
        self.transform = transform

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        try:
            file_path = self.file_paths[idx]
            label = self.labels[idx]

            # 加载音频文件
            waveform, sr = librosa.load(file_path, sr=self.sample_rate)

            # 提取MFCC特征
            mfcc = librosa.feature.mfcc(
                y=waveform,
                sr=self.sample_rate,
                n_mfcc=self.n_mfcc,  # 生成64个MFCC系数
                n_fft=self.n_fft,
                hop_length=self.hop_length
            )
            mfcc = (mfcc - mfcc.mean()) / mfcc.std()  # 归一化处理

            # 填充或裁剪到固定长度
            if mfcc.shape[1] < self.max_length:
                pad_width = self.max_length - mfcc.shape[1]
                mfcc = np.pad(mfcc, ((0, 0), (0, pad_width)), mode='constant')
            else:
                mfcc = mfcc[:, :self.max_length]

            # 转换为张量，确保它是 3D 张量 [channels, height, width]
            mfcc_tensor = torch.from_numpy(np.array(mfcc, dtype=np.float32)).unsqueeze(0)  # 添加通道维度，成为 [1, 64, 431]

            if self.transform:
                mfcc_tensor = self.transform(mfcc_tensor)

            return mfcc_tensor, label
            
        except Exception as e:
            # print(f"Warning: Corrupted file skipped: {self.file_paths[idx]} - {e}")
            # If a file is corrupted, pick a random other index or simply the next index
            return self.__getitem__((idx + 1) % len(self.file_paths))

    def mels(self, file_path):
        # 加载音频文件
        waveform, sr = librosa.load(file_path, sr=self.sample_rate)

        # 提取MFCC特征
        mfcc = librosa.feature.mfcc(
            y=waveform,
            sr=self.sample_rate,
            n_mfcc=self.n_mfcc,
            n_fft=self.n_fft,
            hop_length=self.hop_length
        )
        mfcc = (mfcc - mfcc.mean()) / mfcc.std()  # 归一化处理

        # 填充或裁剪到固定长度
        if mfcc.shape[1] < self.max_length:
            pad_width = self.max_length - mfcc.shape[1]
            mfcc = np.pad(mfcc, ((0, 0), (0, pad_width)), mode='constant')
        else:
            mfcc = mfcc[:, :self.max_length]

        # 转换为张量，确保它是 3D 张量 [channels, height, width]
        mfcc_tensor = torch.from_numpy(np.array(mfcc, dtype=np.float32)).unsqueeze(0)  # 添加通道维度，成为 [1, 64, 431]

        if self.transform:
            mfcc_tensor = self.transform(mfcc_tensor)
        return mfcc_tensor


def train(model, device, train_loader, optimizer, epoch, criterion):
    """
    Train the model for one epoch

    Args:
        model: the neural network model
        device: cuda or cpu
        train_loader: DataLoader for training data
        optimizer: the optimizer to use (e.g. SGD)
        epoch: the current epoch
        criterion: the loss function (e.g. CrossEntropy)

    Returns:
        avg_loss: the average loss over the training set
    """

    model.to(device)
    model.train()
    train_loss = 0
    # Process the images in batches
    for batch_idx, (data, target) in enumerate(tqdm(train_loader)):
        # Recall that GPU is optimized for the operations we are dealing with
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(data)

        loss = criterion(output, target)
        train_loss += loss.item()

        loss.backward()
        optimizer.step()

    # return average loss for the epoch
    avg_loss = train_loss / (batch_idx + 1)
    # print('Training set: Average loss: {:.6f}'.format(avg_loss))
    return avg_loss


def test(model, device, test_loader, criterion):
    """
    Test the model

    Args:
        model: the neural network model
        device: cuda or cpu
        test_loader: DataLoader for test data
        criterion: the loss function (e.g. CrossEntropy)

    Returns:
        test_loss: the average loss over the test set
        accuracy: the accuracy of the model on the test set
        precision: the precision of the model on the test set
        recall: the recall of the model on the test set
        f1: the f1 score of the model on the test set
    """
    model.eval()
    test_loss = 0
    correct = 0
    all_targets = []
    all_predictions = []

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)

            # Get the predicted classes for this batch
            output = model(data)

            # Calculate the loss for this batch
            test_loss += criterion(output, target).item()

            # Calculate the accuracy for this batch
            _, predicted = torch.max(output.data, 1)
            correct += (target == predicted).sum().item()

            # Collect all targets and predictions for metric calculations
            all_targets.extend(target.view_as(predicted).cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

    # Calculate overall metrics
    precision = precision_score(all_targets, all_predictions, average='macro')
    recall = recall_score(all_targets, all_predictions, average='macro')
    f1 = f1_score(all_targets, all_predictions, average='macro')

    # Normalize test loss
    test_loss /= len(test_loader.dataset)
    accuracy = correct / len(test_loader.dataset)

    # print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%), Precision: {:.2f}, Recall: {:.2f}, F1 Score: {:.2f}\n'.format(
    #     test_loss, correct, len(test_loader.dataset), accuracy, precision, recall, f1))

    return test_loss, accuracy, precision, recall, f1


def train_and_test_models(model, device, train_loader, test_loader, optimizer, criterion, epochs, scheduler,
                          start_epoch):
    """
    Train and test the model

    Args:
        model: the neural network model
        device: cuda or cpu
        train_loader: DataLoader for training data
        test_loader: DataLoader for test data
        optimizer: the optimizer to use (e.g. SGD)
        criterion: the loss function (e.g. CrossEntropy)
        epochs: the number of epochs to train
        scheduler: the learning rate scheduler

    Returns:
        all_train_loss: a list of the average training loss for each epoch
        all_test_loss: a list of the average test loss for each epoch
        all_test_accuracy: a list of the accuracy for each epoch
        all_test_precision: a list of the precision for each epoch
        all_test_recall: a list of the recall for each epoch
        all_test_f1: a list of the f1 score for each epoch
    """
    # Track metrics
    all_train_loss = []
    all_test_loss = []
    all_test_accuracy = []
    all_test_precision = []
    all_test_recall = []
    all_test_f1 = []

    for epoch in range(start_epoch + 1, epochs + 1):
        # Train the model
        train_loss = train(model, device, train_loader, optimizer, epoch, criterion)
        all_train_loss.append(train_loss)

        # Test the model
        test_loss, test_accuracy, test_precision, test_recall, test_f1 = test(model, device, test_loader, criterion)
        all_test_loss.append(test_loss)
        all_test_accuracy.append(test_accuracy)
        all_test_precision.append(test_precision)
        all_test_recall.append(test_recall)
        all_test_f1.append(test_f1)

        print(
            f'End of Epoch {epoch}: Train Loss: {train_loss:.6f}, Test Loss: {test_loss:.4f}, Accuracy: {test_accuracy:.2%}')
        # 保存模型
        save_checkpoint(model, optimizer, epoch, test_loss, checkpoint_dir)
        scheduler.step()

    # 在训练循环结束后添加以下代码来保存模型
    model_save_path = f'model_KKAN_Convolutional_Network_epoch_{epochs}.pth'
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")

    model.all_test_accuracy = all_test_accuracy
    model.all_test_precision = all_test_precision
    model.all_test_f1 = all_test_f1
    model.all_test_recall = all_test_recall

    return all_train_loss, all_test_loss, all_test_accuracy, all_test_precision, all_test_recall, all_test_f1


def load_dataset_files(root_dir=" "):
    """加载原始数据文件，并将其打标签分类"""

    # 定义类别及其对应的标签
    class_names = [
        'Piano', 'Violin', 'Guitar', 'Cello', 'Saxophone', 'Flute', 'Oboe',
        'Clarinet', 'Trumpet', 'Trombone', 'French Horn', 'Double Bass',
        'Harp', 'Harmonica', 'Accordion', 'Organ', 'Marimba', 'Vibraphone',
        'Celesta', 'Xylophone'
    ]
    class_to_idx = {class_name: idx for idx, class_name in enumerate(class_names)}

    # 初始化文件路径和标签列表
    file_paths = []
    labels = []

    # 遍历每个类别文件夹
    for class_name in class_names:
        class_dir = os.path.join(root_dir, class_name)
        for fname in os.listdir(class_dir):
            if fname.endswith(".mp3"):  # 只处理 .mp3 文件
                file_paths.append(os.path.join(class_dir, fname))
                labels.append(class_to_idx[class_name])

    # 输出一下检查
    print(f"Total audio files: {len(file_paths)}")
    print(f"Sample file path: {file_paths[0]}")
    print(f"Sample label: {labels[0]}")

    # file_paths = file_paths[:500]  # TODO 测试只取500个样本
    # labels = labels[:500]
    # 将数据集划分为训练集和测试集
    train_paths, test_paths, train_labels, test_labels = train_test_split(file_paths, labels, test_size=0.2,
                                                                          random_state=42)

    # 打印划分后的一些信息
    print(f"Training samples: {len(train_paths)}")
    print(f"Testing samples: {len(test_paths)}")
    return train_paths, test_paths, train_labels, test_labels


if __name__ == '__main__':
    # 验证当前工作目录
    current_directory = os.getcwd()
    print(f"当前工作目录: {current_directory}")

    # 定义数据集根目录
    train_paths, test_paths, train_labels, test_labels = load_dataset_files("ICKAN_Dataset")

    # 创建训练集和测试集的数据集对象
    train_dataset = AudioDataset_MFCC(train_paths, train_labels, n_mels=64, max_length=431)
    test_dataset = AudioDataset_MFCC(test_paths, test_labels, n_mels=64, max_length=431)

    # 创建 DataLoader 对象
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32*2, shuffle=False)

    # 保存路径

    checkpoint_path = None  # 完整路径，如果为None则是不加载
    # checkpoint_path = "/content/drive/MyDrive/Convolutional-KANs-master/checkpoint/checkpoint_epoch_10.pth"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    from models.SimpleModels import SimpleCNN, SimpleCNN_2, SimpleLinear
    from models.ConvNet import ConvNet
    from models.conv_and_kan import ICKAN
    from models.baseline import ConvComp

    model_dict = {
        "cnn_small_mfcc": SimpleCNN,
        "cnn_Medium_mfcc": SimpleCNN_2,
        "mlp_mfcc": SimpleLinear,
        "ConvNet_mfcc": ConvNet,
        "baseline_mfcc": ConvComp,
        "ICKAN_mfcc": ICKAN,
    }

    for model_name in model_dict:
        checkpoint_dir = f"checkpoint/{model_name}"
        os.makedirs(checkpoint_dir, exist_ok=True)
        # 初始化模型
        model_KKAN_Convolutional_Network = model_dict[model_name]()
        model_KKAN_Convolutional_Network.to(device)

        # 定义优化器、损失函数和学习率调度器
        optimizer_KKAN_Convolutional_Network = optim.AdamW(model_KKAN_Convolutional_Network.parameters(), lr=1e-3,
                                                           weight_decay=1e-4)
        scheduler_KKAN_Convolutional_Network = optim.lr_scheduler.ExponentialLR(optimizer_KKAN_Convolutional_Network,
                                                                                gamma=0.8)
        criterion_KKAN_Convolutional_Network = nn.CrossEntropyLoss()

        start_epoch = 0
        if checkpoint_path is not None:
            start_epoch, loss = load_checkpoint(model_KKAN_Convolutional_Network, optimizer_KKAN_Convolutional_Network,
                                                checkpoint_path)
            print(f'load from {checkpoint_path}')

        # 训练和测试模型
        all_train_loss_KKAN_Convolutional_Network, \
            all_test_loss_KKAN_Convolutional_Network, \
            all_test_accuracy_KKAN_Convolutional_Network, \
            all_test_precision_KKAN_Convolutional_Network, \
            all_test_recall_KKAN_Convolutional_Network, \
            all_test_f1_KKAN_Convolutional_Network = train_and_test_models(
            model_KKAN_Convolutional_Network,
            device,
            train_loader,
            test_loader,
            optimizer_KKAN_Convolutional_Network,
            criterion_KKAN_Convolutional_Network,
            epochs=10,
            scheduler=scheduler_KKAN_Convolutional_Network,
            start_epoch=start_epoch
        )

        # 保存训练过程
        result_dict = {
            "name": model_name,
            "model_struct": str(model_KKAN_Convolutional_Network),
            'train_loss': all_train_loss_KKAN_Convolutional_Network,
            'test_loss': all_test_loss_KKAN_Convolutional_Network,
            'test_accuracy': all_test_accuracy_KKAN_Convolutional_Network,
            'test_precision': all_test_precision_KKAN_Convolutional_Network,
            'test_recall': all_test_recall_KKAN_Convolutional_Network,
            'test_f1': all_test_f1_KKAN_Convolutional_Network
        }

        result_saved_path = os.path.join('results', model_name)
        os.makedirs(result_saved_path, exist_ok=True)
        with open(os.path.join(result_saved_path,str(time.time())), 'w', encoding='utf-8') as f:
            json.dump(result_dict, f, indent=4, ensure_ascii=False)
