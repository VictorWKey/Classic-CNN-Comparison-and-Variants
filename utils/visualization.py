import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

def plot_training_curves(history, model_name, save_path=None):
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Loss curves
    ax1.plot(epochs, history['train_loss'], 'b-', label='Train Loss')
    ax1.plot(epochs, history['val_loss'], 'r-', label='Val Loss')
    ax1.set_title(f'{model_name} - Loss')
    ax1.set_xlabel('Época')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Accuracy curves
    ax2.plot(epochs, history['train_acc'], 'b-', label='Train Acc')
    ax2.plot(epochs, history['val_acc'], 'r-', label='Val Acc')
    ax2.set_title(f'{model_name} - Accuracy')
    ax2.set_xlabel('Época')
    ax2.set_ylabel('Accuracy (%)')
    ax2.legend()
    ax2.grid(True)
    
    # F1 score
    ax3.plot(epochs, history['val_f1'], 'g-', label='Val F1')
    ax3.set_title(f'{model_name} - F1 Score')
    ax3.set_xlabel('Época')
    ax3.set_ylabel('F1 Score (%)')
    ax3.legend()
    ax3.grid(True)
    
    # Combined accuracy
    ax4.plot(epochs, history['train_acc'], 'b-', label='Train')
    ax4.plot(epochs, history['val_acc'], 'r-', label='Validation')
    ax4.set_title(f'{model_name} - Train vs Val Accuracy')
    ax4.set_xlabel('Época')
    ax4.set_ylabel('Accuracy (%)')
    ax4.legend()
    ax4.grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()

def plot_confusion_matrix(conf_matrix, class_names, model_name, save_path=None):
    plt.figure(figsize=(10, 8))
    
    # Normalize confusion matrix
    conf_matrix_norm = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis]
    
    sns.heatmap(conf_matrix_norm, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    
    plt.title(f'Matriz de Confusión - {model_name}')
    plt.xlabel('Predicción')
    plt.ylabel('Real')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()

def create_results_table(results_dict):
    df = pd.DataFrame(results_dict)
    return df

def plot_model_comparison(results_df, save_path=None):
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    models = results_df['Modelo']
    
    # Accuracy comparison
    ax1.bar(models, results_df['Test Acc'], color='skyblue', alpha=0.7)
    ax1.set_title('Comparación de Accuracy')
    ax1.set_ylabel('Test Accuracy (%)')
    ax1.tick_params(axis='x', rotation=45)
    ax1.grid(True, alpha=0.3)
    
    # F1 comparison
    ax2.bar(models, results_df['Test F1'], color='lightgreen', alpha=0.7)
    ax2.set_title('Comparación de F1 Score')
    ax2.set_ylabel('Test F1 (%)')
    ax2.tick_params(axis='x', rotation=45)
    ax2.grid(True, alpha=0.3)
    
    # Parameters comparison
    ax3.bar(models, results_df['Params (M)'], color='orange', alpha=0.7)
    ax3.set_title('Número de Parámetros')
    ax3.set_ylabel('Parámetros (M)')
    ax3.tick_params(axis='x', rotation=45)
    ax3.grid(True, alpha=0.3)
    
    # Time comparison
    ax4.bar(models, results_df['t/epoca (s)'], color='salmon', alpha=0.7)
    ax4.set_title('Tiempo por Época')
    ax4.set_ylabel('Tiempo (s)')
    ax4.tick_params(axis='x', rotation=45)
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()

def plot_combined_training_curves(histories_dict, save_path=None):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Loss curves
    for model_name, history in histories_dict.items():
        epochs = range(1, len(history['train_loss']) + 1)
        ax1.plot(epochs, history['val_loss'], label=f'{model_name}')
    
    ax1.set_title('Comparación de Curvas de Pérdida (Validación)')
    ax1.set_xlabel('Época')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Accuracy curves
    for model_name, history in histories_dict.items():
        epochs = range(1, len(history['val_acc']) + 1)
        ax2.plot(epochs, history['val_acc'], label=f'{model_name}')
    
    ax2.set_title('Comparación de Curvas de Accuracy (Validación)')
    ax2.set_xlabel('Época')
    ax2.set_ylabel('Accuracy (%)')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()