"""
Configuración principal para el proyecto de comparación de CNNs
"""

# Configuración de entrenamiento (protocolo común)
TRAINING_CONFIG = {
    'epochs': 60,
    'batch_size': 64,
    'optimizer': 'SGD',
    'learning_rate': 0.001,
    'momentum': 0.9,
    'weight_decay': 5e-4,
    'criterion': 'CrossEntropyLoss'
}

# Configuración de datos
DATA_CONFIG = {
    'data_dir': 'data',
    'input_size': 224,
    'train_split': 0.8,
    'val_split': 0.1,
    'test_split': 0.1,
    'seed': 42
}

# Rutas de archivos
PATHS = {
    'models': 'models',
    'utils': 'utils', 
    'notebooks': 'notebooks',
    'results': 'results',
    'checkpoints': 'checkpoints',
    'data': 'data'
}

# Modelos a entrenar
MODELS = {
    'customlenet': {
        'base': 'CustomLenet',
        'variants': ['CustomLenet_NoBN', 'CustomLenet_NoDropout']
    },
    'alexnet': {
        'base': 'AlexNet',
        'variants': ['AlexNet_BN', 'AlexNet_Reduced']
    },
    'vgg16': {
        'base': 'VGG16',
        'variants': ['VGG16_BN', 'VGG16_GAP']
    }
}

# Métricas a calcular
METRICS = ['accuracy', 'f1_macro', 'confusion_matrix']

# Configuración de visualización
VIZ_CONFIG = {
    'dpi': 300,
    'figsize': (12, 8),
    'save_format': 'png'
}