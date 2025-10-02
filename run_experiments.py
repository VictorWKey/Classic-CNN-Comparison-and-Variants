#!/usr/bin/env python3
"""
Script principal para ejecutar todos los experimentos de comparación de CNNs
"""

import os
import subprocess

def run_notebook(notebook_path):
    """Ejecuta un notebook de Jupyter"""
    print(f"Ejecutando {notebook_path}...")
    result = subprocess.run([
        'jupyter', 'nbconvert', 
        '--to', 'notebook',
        '--execute',
        '--inplace',
        notebook_path
    ], capture_output=True, text=True)
    
    if result.returncode == 0:
        print(f"✓ {notebook_path} ejecutado exitosamente")
    else:
        print(f"✗ Error ejecutando {notebook_path}")
        print(result.stderr)
        return False
    return True

def main():
    print("="*60)
    print("COMPARACIÓN DE CNNs CLÁSICAS - EJECUCIÓN AUTOMÁTICA")
    print("="*60)
    
    # Verificar estructura de carpetas
    required_dirs = ['models', 'utils', 'notebooks', 'results', 'checkpoints', 'data']
    for dir_name in required_dirs:
        if not os.path.exists(dir_name):
            print(f"Creando directorio: {dir_name}")
            os.makedirs(dir_name)
    
    # Lista de notebooks a ejecutar en orden
    notebooks = [
        'notebooks/01_LeNet_Training.ipynb',
        'notebooks/02_AlexNet_Training.ipynb', 
        'notebooks/03_VGG16_Training.ipynb',
        'notebooks/04_Comparative_Analysis.ipynb'
    ]
    
    # Verificar que existen los datos o descargarlos
    if not os.path.exists('data') or not os.listdir('data'):
        print("⚠️  No se encontraron datos en 'data/'. Descargando CIFAR10...")
        import torchvision
        import torchvision.transforms as transforms

        if not os.path.exists("data"):
            os.makedirs("data")

        torchvision.datasets.CIFAR10(
            root="data", train=True, download=True, transform=transforms.ToTensor()
        )
        torchvision.datasets.CIFAR10(
            root="data", train=False, download=True, transform=transforms.ToTensor()
        )

        print("✅ CIFAR10 descargado en 'data/'")
    else:
        print(f"Encontrados {len(os.listdir('data'))} clases en el dataset")
    
    # Ejecutar notebooks secuencialmente
    success_count = 0
    for notebook in notebooks:
        if os.path.exists(notebook):
            if run_notebook(notebook):
                success_count += 1
        else:
            print(f"⚠️  Notebook no encontrado: {notebook}")
    
    print("\n" + "="*60)
    print(f"RESUMEN: {success_count}/{len(notebooks)} notebooks ejecutados exitosamente")
    
    if success_count == len(notebooks):
        print("✓ Todos los experimentos completados")
        print("📊 Revisa los resultados en:")
        print("   - results/: Tablas, figuras y reportes")
        print("   - checkpoints/: Modelos entrenados")
    else:
        print("⚠️  Algunos experimentos fallaron. Revisa los errores arriba.")
    
    print("="*60)

if __name__ == "__main__":
    main()
