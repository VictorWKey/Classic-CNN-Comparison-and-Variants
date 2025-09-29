# Propuesta de Variantes para Arquitecturas CNN

## Introducción

Este documento describe las variantes propuestas para cada arquitectura CNN clásica (LeNet, AlexNet, VGG16) y las hipótesis teóricas que sustentan cada modificación. Las variantes fueron diseñadas para evaluar el impacto de diferentes técnicas de normalización, regularización y capacidad del modelo.

## Metodología de Selección de Variantes

Las variantes fueron seleccionadas siguiendo tres criterios principales:

1. **Impacto teórico fundamentado**: Cada modificación tiene base en literatura científica
2. **Modificación atómica**: Una sola modificación por variante para aislar el efecto
3. **Relevancia práctica**: Técnicas ampliamente utilizadas en deep learning moderno

## Variantes por Arquitectura

### CustomLenet

**Modelo Base**: CustomLenet ya incluye BatchNorm tras cada convolución, BatchNorm en FC, y Dropout para regularización. Es una versión modernizada de LeNet con 32→64 canales.

#### Variante 1: CustomLenet sin BatchNorm (CustomLenet_NoBN)
- **Modificación**: Eliminación de todas las capas BatchNorm (conv y FC)
- **Justificación técnica**: 
  - Evaluar el impacto específico de BatchNorm en estabilidad y convergencia
  - Aislar el efecto de normalización vs. otros componentes (Dropout)
  - Comparar con entrenamiento "clásico" sin normalización
- **Hipótesis**: La ausencia de BatchNorm resultará en convergencia más lenta e inestable, mayor sensibilidad a inicialización
- **Expectativa**: Peor rendimiento y curvas de entrenamiento más ruidosas

#### Variante 2: CustomLenet sin Dropout (CustomLenet_NoDropout)
- **Modificación**: Eliminación de Dropout en capas fully connected
- **Justificación técnica**:
  - Evaluar el impacto de Dropout como regularizador
  - Analizar el trade-off entre capacidad de memorización y generalización
  - Medir susceptibilidad al overfitting sin regularización explícita
- **Hipótesis**: La ausencia de Dropout causará overfitting, especialmente visible en diferencia train-validation
- **Expectativa**: Mejor rendimiento en entrenamiento pero peor generalización

### AlexNet

#### Variante 1: AlexNet + BatchNorm (AlexNet_BN)
- **Modificación**: BatchNorm2d tras todas las capas convolucionales
- **Justificación técnica**:
  - AlexNet original susceptible a gradientes inestables por su profundidad
  - BatchNorm estabiliza el entrenamiento en redes profundas
  - Reduce dependencia de inicialización de pesos
- **Hipótesis**: BatchNorm reducirá el overfitting y acelerará la convergencia de AlexNet
- **Expectativa**: Curvas de entrenamiento más suaves y mejor generalización

#### Variante 2: AlexNet Reduced (AlexNet_Reduced)
- **Modificación**: Reducción de capas FC (3→2 capas, 4096→2048 neuronas)
- **Justificación técnica**:
  - Las capas FC concentran la mayoría de parámetros (~90% en AlexNet)
  - Reducción de parámetros disminuye overfitting
  - Menor costo computacional y memoria
- **Hipótesis**: La reducción de FC mantendrá el rendimiento mientras reduce overfitting y tiempo de entrenamiento
- **Expectativa**: Modelo más eficiente con rendimiento comparable

### VGG16

#### Variante 1: VGG16 + BatchNorm (VGG16_BN)
- **Modificación**: BatchNorm2d tras cada capa convolucional
- **Justificación técnica**:
  - VGG16 es muy profunda (16 capas), propensa a vanishing gradients
  - BatchNorm es crítica para entrenar redes muy profundas efectivamente
  - Permite prescindir de pre-entrenamiento
- **Hipótesis**: BatchNorm será esencial para que VGG16 entrene efectivamente desde cero
- **Expectativa**: Diferencia significativa en convergencia vs. modelo base

#### Variante 2: VGG16 + Global Average Pooling (VGG16_GAP)
- **Modificación**: Reemplazo de capas FC densas por Global Average Pooling
- **Justificación técnica**:
  - GAP reduce parámetros de ~134M a ~15M
  - Mantiene información espacial hasta el final
  - Menos propenso a overfitting que FC densas
  - Inspirado en arquitecturas modernas (ResNet, etc.)
- **Hipótesis**: GAP reducirá drásticamente los parámetros manteniendo el rendimiento
- **Expectativa**: Modelo mucho más eficiente con rendimiento competitivo

## Preguntas de Investigación

### Pregunta Principal
¿Cómo impactan las técnicas de normalización, regularización y modificaciones de capacidad en el rendimiento de arquitecturas CNN clásicas?

### Preguntas Específicas

1. **Normalización**: ¿BatchNorm mejora consistentemente todas las arquitecturas independientemente de su profundidad?

2. **Capacidad**: ¿El incremento de parámetros siempre mejora el rendimiento o existe un trade-off con overfitting?

3. **Regularización**: ¿GAP puede reemplazar efectivamente las capas FC manteniendo el rendimiento?

4. **Eficiencia**: ¿Qué variante ofrece el mejor balance rendimiento/costo computacional?

## Hipótesis Generales

### H1: Normalización
BatchNorm será crucial para estabilidad y convergencia. CustomLenet sin BN mostrará convergencia más lenta y ruidosa comparado con la versión base.

### H2: Regularización
Dropout será esencial para prevenir overfitting. CustomLenet sin Dropout mostrará mejor rendimiento en entrenamiento pero peor generalización.

### H3: Modernización de Arquitecturas
Las técnicas modernas (BatchNorm, GAP) pueden modernizar efectivamente arquitecturas clásicas, mejorando su rendimiento y eficiencia.

### H4: Trade-off Rendimiento-Eficiencia
Las variantes mostrarán diferentes posiciones en el espacio rendimiento-eficiencia, permitiendo selección según restricciones del problema.

## Métricas de Evaluación

### Primarias
- **Accuracy**: Medida principal de rendimiento
- **Macro-F1**: Rendimiento balanceado entre clases
- **Pérdida de validación**: Indicador de overfitting

### Secundarias  
- **Número de parámetros**: Complejidad del modelo
- **Tiempo por época**: Eficiencia computacional
- **Convergencia**: Épocas hasta estabilización

### Análisis Comparativo
- **Curvas de entrenamiento**: Comportamiento durante el entrenamiento
- **Matrices de confusión**: Errores por clase
- **Análisis costo-beneficio**: Rendimiento vs. recursos

## Protocolo Experimental

### Controles
- **Dataset**: Mismo conjunto de datos para todas las variantes
- **Partición**: 80/10/10 (train/val/test) con semilla fija
- **Hiperparámetros**: SGD, lr=0.001, momentum=0.9, weight_decay=5e-4
- **Épocas**: 60 para todas las variantes
- **Hardware**: Mismo dispositivo para comparación justa

### Variables
- **Arquitectura**: Diferentes modelos base y variantes
- **Inicialización**: Cada modelo inicializado independientemente

Este enfoque sistemático permitirá evaluar objetivamente el impacto de cada modificación y generar conclusiones fundamentadas sobre la efectividad de las técnicas aplicadas a arquitecturas CNN clásicas.