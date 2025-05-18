import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader, random_split
from PIL import Image
import os
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Решение для Windows multiprocessing
if __name__ == '__main__':
    # Установка переменной окружения
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

    # Функция для проверки изображений
    def validate_images(folder_path):
        valid_images = []
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                    file_path = os.path.join(root, file)
                    try:
                        with Image.open(file_path) as img:
                            img.verify()
                        valid_images.append(file_path)
                    except (IOError, SyntaxError) as e:
                        print(f"Удалено поврежденное изображение: {file_path} ({e})")
        return valid_images

    # Путь к данным
    data_dir = './SportsEquipment'
    print("Проверка целостности изображений...")
    valid_images = validate_images(data_dir)

    # Проверка, что найдены изображения
    if not valid_images:
        raise ValueError("Не найдено ни одного валидного изображения в папке SportsEquipment!")

    # Трансформации данных
    data_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Кастомный датасет
    class SportsEquipmentDataset(torch.utils.data.Dataset):
        def __init__(self, image_paths, transform=None):
            self.image_paths = image_paths
            self.transform = transform
            self.classes = ['barbells', 'dumbbells', 'machines']
            self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}
            
        def __len__(self):
            return len(self.image_paths)
        
        def __getitem__(self, idx):
            img_path = self.image_paths[idx]
            try:
                image = Image.open(img_path).convert('RGB')
                label_name = os.path.basename(os.path.dirname(img_path))
                
                if label_name not in self.class_to_idx:
                    raise ValueError(f"Неизвестный класс: {label_name}")
                
                label = self.class_to_idx[label_name]
                
                if self.transform:
                    image = self.transform(image)
                    
                return image, label
            except Exception as e:
                print(f"Ошибка загрузки {img_path}: {e}")
                # Возвращаем случайное другое изображение вместо проблемного
                return self[(idx + 1) % len(self)]

    # Создание и разделение датасета
    dataset = SportsEquipmentDataset(valid_images, data_transforms)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    # DataLoader'ы с num_workers=0
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0)

    # Модель
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 3)  # 3 класса

    # Устройство
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Функция потерь и оптимизатор
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    # Обучение (улучшенное)
    def train_model(model, criterion, optimizer, num_epochs=10):
        losses = []
        for epoch in range(num_epochs):
            model.train()
            running_loss = 0.0
            correct = 0
            total = 0
            
            for inputs, labels in train_loader:
                if inputs is None:
                    continue
                    
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            
            epoch_loss = running_loss / len(train_loader)
            epoch_acc = 100 * correct / total
            losses.append(epoch_loss)
            print(f'Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.2f}%')
        
        return losses

    # Запуск обучения
    print("Начало обучения...")
    losses = train_model(model, criterion, optimizer, num_epochs=10)

    # Визуализация потерь
    plt.figure(figsize=(10, 5))
    plt.plot(losses)
    plt.title('График потерь при обучении')
    plt.xlabel('Эпохи')
    plt.ylabel('Потери')
    plt.grid(True)
    plt.show()

    # Тестирование (улучшенное)
    def evaluate(model, test_loader):
        model.eval()
        correct = 0
        total = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for inputs, labels in test_loader:
                if inputs is None:
                    continue
                    
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        accuracy = 100 * correct / total
        print(f'Точность на тестовых данных: {accuracy:.2f}%')
        return all_labels, all_preds

    # Оценка модели
    print("Оценка модели...")
    true_labels, pred_labels = evaluate(model, test_loader)

    # Матрица ошибок
    def plot_confusion_matrix(true_labels, pred_labels, class_names):
        cm = np.zeros((len(class_names), len(class_names)), dtype=int)
        for t, p in zip(true_labels, pred_labels):
            cm[t][p] += 1
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=class_names,
                    yticklabels=class_names)
        plt.xlabel('Предсказанные')
        plt.ylabel('Истинные')
        plt.title('Матрица ошибок')
        plt.show()

    # Визуализация матрицы ошибок
    plot_confusion_matrix(true_labels, pred_labels, dataset.classes)