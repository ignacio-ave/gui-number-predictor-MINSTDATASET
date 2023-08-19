# Documentación del programa: DigitsPredictor

## Descripción general:
Este programa implementa una red neuronal simple con una capa oculta para predecir dígitos manuscritos basados en el conjunto de datos MNIST. Se proporciona una interfaz gráfica para que los usuarios dibujen un dígito y luego predecir cuál es ese dígito usando la red neuronal entrenada.

## Librerías utilizadas:

- **numpy**: Para operaciones matriciales y computación numérica.
- **struct**: Para desempaquetar los bytes leídos de los archivos MNIST.
- **tkinter**: Para crear la interfaz gráfica de usuario.
- **PIL (Image, ImageDraw, ImageFilter)**: Para manipular imágenes en la interfaz gráfica.
- **matplotlib.pyplot**: Para visualizaciones (aunque no se utiliza activamente en el programa).

## Constantes:

- **INPUT_SIZE**: Tamaño de entrada (784 para MNIST ya que las imágenes son de 28x28).
- **HIDDEN_SIZE**: Tamaño de la capa oculta.
- **OUTPUT_SIZE**: Tamaño de salida (10 para dígitos del 0 al 9).
- **RANDOM_SEED**: Semilla para inicialización de pesos.
- **NUM_CLASSES**: Número de clases (10 para dígitos del 0 al 9).

## Funciones:

### Carga de datos MNIST:
- **load_images(filename)**: Carga las imágenes del conjunto de datos MNIST desde un archivo dado.
- **load_labels(filename)**: Carga las etiquetas del conjunto de datos MNIST desde un archivo dado.

### Inicialización y definiciones de la red neuronal:
- **initialize_network_parameters()**: Inicializa los pesos y sesgos de la red neuronal.
- **relu(Z)**: Función de activación ReLU.
- **relu_derivative(Z)**: Derivada de la función ReLU.
- **softmax(Z)**: Función Softmax para obtener probabilidades.
- **cross_entropy_loss(Y_true, Y_pred)**: Función de pérdida de entropía cruzada.
- **cross_entropy_loss_derivative(Y_true, Y_pred)**: Derivada de la función de pérdida de entropía cruzada.
- **forward_propagation(X)**: Realiza la propagación hacia adelante a través de la red.
- **backward_propagation(X, Y, Z1, A1, Z2, A2)**: Realiza la propagación hacia atrás y calcula los gradientes.
- **update_parameters(dW1, db1, dW2, db2, learning_rate)**: Actualiza los pesos y sesgos usando los gradientes.
- **train_neural_network(X, Y, learning_rate=0.1, epochs=100)**: Entrena la red neuronal.
- **save_weights(filename='weights.npz')**: Guarda los pesos y sesgos de la red en un archivo.
- **load_weights(filename='weights.npz')**: Carga los pesos y sesgos de la red desde un archivo.

### Clase DigitsPredictor:

Esta clase crea y gestiona la interfaz gráfica de usuario que permite a los usuarios dibujar un dígito y predecirlo.

#### Métodos:

- **__init__(self, root)**: Inicializa la interfaz gráfica y configura el lienzo y los botones.
- **start_pos(self, event)**: Establece la posición inicial del trazo del lápiz.
- **paint(self, event)**: Dibuja en el lienzo según el movimiento del ratón.
- **use_pen(self)**: Configura el lápiz para dibujar.
- **use_eraser(self)**: Configura el lápiz como goma de borrar.
- **predict_digit(self)**: Predice el dígito dibujado por el usuario.
- **clear_canvas(self)**: Limpia el lienzo.

## Flujo principal del programa:

1. Se solicita al usuario que ingrese la ruta de la carpeta de la base de datos MNIST.
2. Se carga el conjunto de datos MNIST.
3. Se le pide al usuario que elija entre entrenar la red neuronal o cargar pesos existentes.
4. Si se elige entrenar, la red se entrena y luego se guardan los pesos.
5. Si se elige cargar, se intenta cargar los pesos desde un archivo. Si el archivo no existe, se entrena la red.
6. Se inicia la interfaz gráfica de usuario para que los usuarios puedan dibujar y predecir dígitos.


## Créditos

Los datos utilizados en este proyecto provienen del conjunto de datos MNIST. La citación completa es:

> LeCun, Y., Cortes, C., & Burges, C. J. (2010). MNIST handwritten digit database. AT&T Labs [Online]. Available: [http://yann.lecun.com/exdb/mnist](http://yann.lecun.com/exdb/mnist)
