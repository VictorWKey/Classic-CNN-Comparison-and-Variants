import os
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image
import random

from torchvision.datasets import CIFAR10


class CustomImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = sorted([d for d in os.listdir(root_dir)
                               if os.path.isdir(os.path.join(root_dir, d))])
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}

        self.samples = []
        for class_name in self.classes:
            class_dir = os.path.join(root_dir, class_name)
            class_idx = self.class_to_idx[class_name]

            for img_name in os.listdir(class_dir):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                    img_path = os.path.join(class_dir, img_name)
                    self.samples.append((img_path, class_idx))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, label


def get_transforms(input_size=224, augment=True):
    if augment:
        train_transform = transforms.Compose([
            transforms.Resize((input_size + 32, input_size + 32)),
            transforms.RandomCrop((input_size, input_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
    else:
        train_transform = transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    val_test_transform = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    return train_transform, val_test_transform


def create_data_loaders(data_dir, batch_size=64, input_size=224, train_split=0.7, val_split=0.2, seed=42):
    """
    Crea DataLoaders para entrenamiento, validación y test
    """
    # Verificar qué tipo de dataset tenemos
    has_cifar10_data = os.path.exists(os.path.join(data_dir, 'cifar-10-batches-py'))

    if has_cifar10_data:
        print("🔍 Detectado dataset CIFAR-10")
        return _create_cifar10_loaders(data_dir, batch_size, input_size, train_split, val_split, seed)
    else:
        print("🔍 Detectado dataset personalizado")
        return _create_custom_loaders(data_dir, batch_size, input_size, train_split, val_split, seed)


def _create_cifar10_loaders(data_dir, batch_size=64, input_size=224, train_split=0.7, val_split=0.2, seed=42):
    """Crea DataLoaders específicos para CIFAR-10"""

    # Transformaciones optimizadas para CIFAR-10
    train_transform = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomCrop(input_size, padding=4),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                             std=[0.2023, 0.1994, 0.2010])
    ])

    val_test_transform = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                             std=[0.2023, 0.1994, 0.2010])
    ])

    # Cargar dataset CIFAR-10
    try:
        train_val_dataset = CIFAR10(
            root=data_dir,
            train=True,
            download=True,
            transform=train_transform
        )

        test_dataset = CIFAR10(
            root=data_dir,
            train=False,
            download=True,
            transform=val_test_transform
        )
    except Exception as e:
        print(f"❌ Error al cargar CIFAR-10: {e}")
        return None, None, None, []

    # Dividir train en train y validation
    dataset_size = len(train_val_dataset)
    val_size = int(val_split * dataset_size)
    train_size = dataset_size - val_size

    # Usar random_split con semilla para reproducibilidad
    torch.manual_seed(seed)
    train_dataset, val_dataset = random_split(
        train_val_dataset, [train_size, val_size]
    )

    # Crear DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )

    class_names = train_val_dataset.classes

    print(f"✅ CIFAR-10 cargado:")
    print(f"   - Entrenamiento: {len(train_dataset)} muestras")
    print(f"   - Validación: {len(val_dataset)} muestras")
    print(f"   - Test: {len(test_dataset)} muestras")
    print(f"   - Clases: {class_names}")

    return train_loader, val_loader, test_loader, class_names


def _create_custom_loaders(data_dir, batch_size=64, input_size=224, train_split=0.7, val_split=0.2, seed=42):
    """Crea DataLoaders para datasets personalizados"""

    # Transformaciones para datasets personalizados
    train_transform, val_test_transform = get_transforms(input_size, augment=True)

    # Buscar directorios de train y test
    train_dir = os.path.join(data_dir, 'train')
    test_dir = os.path.join(data_dir, 'test')

    # Si no existe estructura train/test, usar directorio único
    if not os.path.exists(train_dir):
        print("⚠️  No se encontró estructura train/test, usando directorio único")
        full_dataset = CustomImageDataset(root_dir=data_dir, transform=train_transform)

        # Dividir en train/val/test
        dataset_size = len(full_dataset)
        train_size = int(train_split * dataset_size)
        val_size = int(val_split * dataset_size)
        test_size = dataset_size - train_size - val_size

        torch.manual_seed(seed)
        train_dataset, val_dataset, test_dataset = random_split(
            full_dataset, [train_size, val_size, test_size]
        )

        # Aplicar transformaciones diferentes a val y test
        val_dataset.dataset.transform = val_test_transform
        test_dataset.dataset.transform = val_test_transform

        class_names = full_dataset.classes

    else:
        # Estructura train/test existente
        train_val_dataset = CustomImageDataset(root_dir=train_dir, transform=train_transform)
        test_dataset = CustomImageDataset(root_dir=test_dir, transform=val_test_transform)

        # Dividir train en train y validation
        dataset_size = len(train_val_dataset)
        val_size = int(val_split * dataset_size)
        train_size = dataset_size - val_size

        torch.manual_seed(seed)
        train_dataset, val_dataset = random_split(
            train_val_dataset, [train_size, val_size]
        )

        class_names = train_val_dataset.classes

    # Crear DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )

    print(f"✅ Dataset personalizado cargado:")
    print(f"   - Entrenamiento: {len(train_dataset)} muestras")
    print(f"   - Validación: {len(val_dataset)} muestras")
    print(f"   - Test: {len(test_dataset)} muestras")
    print(f"   - Clases: {class_names}")

    return train_loader, val_loader, test_loader, class_names


def get_dataset_info(data_dir, input_size=224):
    """
    Obtiene información del dataset - VERSIÓN SIMPLIFICADA Y CORREGIDA
    """
    try:
        print(f"🔍 Analizando dataset en: {data_dir}")

        # Verificar qué tipo de dataset tenemos
        has_cifar10_data = os.path.exists(os.path.join(data_dir, 'cifar-10-batches-py'))

        if has_cifar10_data:
            print("📊 Procesando CIFAR-10...")
            return _get_cifar10_info_simple(data_dir)
        else:
            print("📊 Procesando dataset personalizado...")
            return _get_custom_dataset_info_simple(data_dir)

    except Exception as e:
        print(f"❌ Error en get_dataset_info: {e}")
        return None


def _get_cifar10_info_simple(data_dir):
    """Obtiene información de CIFAR-10 de forma simple"""
    try:
        # Cargar datasets sin transformaciones complejas
        train_dataset = CIFAR10(root=data_dir, train=True, download=True)
        test_dataset = CIFAR10(root=data_dir, train=False, download=True)

        # Información básica
        class_names = train_dataset.classes
        num_classes = len(class_names)
        total_samples = len(train_dataset) + len(test_dataset)

        # Contar muestras por clase
        samples_per_class = {}
        for class_name in class_names:
            samples_per_class[class_name] = 0

        # Contar en train
        for label in train_dataset.targets:
            class_name = class_names[label]
            samples_per_class[class_name] += 1

        # Contar en test
        for label in test_dataset.targets:
            class_name = class_names[label]
            samples_per_class[class_name] += 1

        # Obtener resolución (CIFAR-10 es 32x32)
        sample_resolution = (32, 32)

        info = {
            "num_classes": num_classes,
            "class_names": class_names,
            "total_samples": total_samples,
            "train_samples": len(train_dataset),
            "test_samples": len(test_dataset),
            "samples_per_class": samples_per_class,
            "sample_resolution": sample_resolution,
            "dataset_type": "CIFAR-10"
        }

        print("✅ Información de CIFAR-10 obtenida correctamente")
        return info

    except Exception as e:
        print(f"❌ Error al obtener información de CIFAR-10: {e}")
        return None


def _get_custom_dataset_info_simple(data_dir):
    """Obtiene información de datasets personalizados de forma simple"""
    try:
        # Verificar estructura de directorios
        train_dir = os.path.join(data_dir, 'train')
        test_dir = os.path.join(data_dir, 'test')

        if os.path.exists(train_dir) and os.path.exists(test_dir):
            # Estructura train/test
            train_dataset = CustomImageDataset(root_dir=train_dir)
            test_dataset = CustomImageDataset(root_dir=test_dir)
            total_samples = len(train_dataset) + len(test_dataset)
            class_names = train_dataset.classes
        elif os.path.exists(train_dir):
            # Solo directorio train
            train_dataset = CustomImageDataset(root_dir=train_dir)
            test_dataset = None
            total_samples = len(train_dataset)
            class_names = train_dataset.classes
        else:
            # Directorio único
            train_dataset = CustomImageDataset(root_dir=data_dir)
            test_dataset = None
            total_samples = len(train_dataset)
            class_names = train_dataset.classes

        if total_samples == 0:
            print("❌ El dataset está vacío")
            return None

        # Información básica
        num_classes = len(class_names)

        # Contar muestras por clase
        samples_per_class = {cls: 0 for cls in class_names}

        # Contar en train
        for _, label in train_dataset.samples:
            samples_per_class[class_names[label]] += 1

        # Contar en test si existe
        if test_dataset:
            for _, label in test_dataset.samples:
                samples_per_class[class_names[label]] += 1

        # Obtener resolución de una muestra
        if len(train_dataset.samples) > 0:
            img_path, _ = train_dataset.samples[0]
            with Image.open(img_path) as img:
                sample_resolution = img.size
        else:
            sample_resolution = (0, 0)

        info = {
            "num_classes": num_classes,
            "class_names": class_names,
            "total_samples": total_samples,
            "samples_per_class": samples_per_class,
            "sample_resolution": sample_resolution,
            "dataset_type": "Custom"
        }

        print("✅ Información del dataset personalizado obtenida correctamente")
        return info

    except Exception as e:
        print(f"❌ Error al obtener información del dataset personalizado: {e}")
        return None