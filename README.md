# Resumen

Esta practica guia al estudiante en el diseno, entrenamiento y evaluacion de tres arquitecturas clasiccas de redes convolucionales (LeNet, AlexNet y VGG16) sobre un problema de clasificacion de imagenes, bajo el enfoque IMRA (Introduccion Metodologia Resultados y Analisis)

## Introducción

### Contexto

Las CNN han demostrado un desempeno sobresaliente en clasificacion de imagenes. En esta practica se comparan 3 arquitecturas, analizando sus caracteristicas como profundidad, tamano receptivo, normalizacion, capacidad) y su relacion con el comportamiento en el entrenamiento.

### Problema

Clasificar imagenes del conjunto de datos de preferencia, con alta exactitud y buena generalizacion, bajo un protocolo comun y con modificaciones controladas (variantes)

### Objetivos de aprendizaje

- Implementar y entrenar las arquitecturas sobre el dataset seleccionado
- Disenar y evaluar variantes (minimo 2 por modelo) modificando capas, regularizacion o entrenamiento cuantificando su impacto.
- Analizar metricas (accuracy, macro-F1) y curvas (perdida y accuracy) y elaborar un informe IMRA reproducible

### Preguntas e hipótesis

Formule preguntas concretas (por ejemplo, BatchNorm mejora AlexNet en el dataset) y una hipotesis, por cada variante propuesta, basada en teoria (normalizacion, capacidad y regularizacion).

## Metodología

### Datos y partición

- Conjunto de datos: describir brevemente clases, tamano, resolucion, usos, aplicaciones y antecedentes.
- Particion: entrenamiento/validacion/prueba (por ejemplo 80/10/10) con semilla fija para reproducibilidad.
- Transformaciones base (train). En validacion/prueba, solo tensor + normalizacion.

### Modelos (base)

- **LeNet (base):** 2 conv + 2 FC; activacion ReLU; MaxPool. Variantes sugeridas: BatchNorm tras conv, incremento de canales.
- **AlexNet (ajustada a N x M de las imagenes):** 5 conv + 3 FC; ReLU; MaxPool; Dropout. Variantes: BatchNorm, reduccion de FC, AdamW.
- **VGG16:** Bloques 3 x 3 + MaxPool. Variantes: VGG16-BN, Global Average Pooling (GAP) en lugar de FC densas, Dropout.

### Protocolo de entrenamiento (común)

Epocas: 60. Batch size: 64, Optimizador: SGD (momentum = 0.9, weight_decay=5e-4). LR inicial: 0.001. Criterion: Cross-Entropy. Semillas: fijadas para reproducibilidad.

### Variantes (objetivo obligatorio)

Cada equipo debe proponer al menos dos variantes por modelo (una modificacion por experimento) y justificar su expectativa de mejora. Ejemplos:

- Arquitectura: BN en LeNet; AlexNet con BN y reduccion de FC; VGG16 con GAP.
- Entrenamiento: cambiar a AdamW, Adam o algun otro optimizador.
- Regularizacion: ajuste de weight decay; Dropout en FC.

### Métricas e histórico

Principales: Accuracy, macro-F1, matriz de confusion. Curvas: perdida y accuracy (train/val). Costo: tiempo por epoca y #parametros por modelo. Guardar checkpoint del mejor modelo por accuracy de validacion.

## Resultados

### Tablas

- **Tabla 1:** Desempeno del modelo base, desempeno y costo:
  Tabla con filas LeNet, AlexNet y VGG16 y columnas Modelo, Params (M), t/epoca (s), Val acc, Val F1, Test Acc y Test F1.

- **Tabla 2:** variantes por modelo: cambio atomico y efecto en metricas.
  Tabla con filas LeNet, AlexNet y VGG16, con columnas Modelo, Variante (Por ejemplo +BN, +BN o GAP), Cambio clave (por ejemplo BN tras conv, BN todas conv o FC -> GAP), Cambio de accuracy, cambio en F1 y observaciones.

### Figuras

- **Figura 1:** Curvas de perdida (entrenamiento/validacion)
- **Figura 2:** Curvas de exactitud (validacion)
- **Figura 3:** Matriz de confusion del mejor modelo en prueba

## Análisis

Discuta si la hipotesis se confirman. Analice: rendimiento vs costo computacional y numero de parametros, errores frecuentes por clase. Proponga mejoras que ayuden a tener un mejor balance entre rendimiento y coste computacional (por ejemplo aumento de datos, balanceo, transfer learning)

## Entregables

- Informe (PDF) con filosofia IMRA con 1-2 paginas por seccion, tablas/figuras y apendice con hardware y tiempos.
- Fijar semillas, documentar versiones y comandos de entrenamiento.
