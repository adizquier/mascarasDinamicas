import os
os.environ["SM_FRAMEWORK"] = "tf.keras"
import segmentation_models as sm
import tensorflow as tf
import numpy as np
from tensorflow.keras.applications.resnet_v2 import ResNet50V2, preprocess_input
from do import GlobalAveragePooling2D, Dense, Input, Concatenate
from tensorflow.keras.layers import Conv2D, MaxPooling2D, UpSampling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.layers import Dense, Dropout, Flatten
from livelossplot.inputs.keras import PlotLossesCallback


BATCH_SIZE = 128
train_data_dir = 'train'
test_data_dir = 'test'

input_shape = (224, 224, 3)
optim_1 = Adam(learning_rate=0.001)
n_classes=5

train_generator = ImageDataGenerator(rotation_range=90, 
                                     brightness_range=[0.1, 0.7],
                                     width_shift_range=0.5, 
                                     height_shift_range=0.5,
                                     horizontal_flip=False, 
                                     vertical_flip=False,
                                     validation_split=0.4,
                                     preprocessing_function=preprocess_input) # ResNet50V2 preprocessing

test_generator = ImageDataGenerator(preprocessing_function=preprocess_input) # ResNet50V2 preprocessing

lot_loss_1 = PlotLossesCallback()

# ModelCheckpoint callback - save best weights
tl_checkpoint_1 = ModelCheckpoint(filepath='Encoder_ResNet50V2.weights.best.keras',
                                  monitor = 'loss',
                                  save_best_only=True,
                                  verbose=1)

# EarlyStopping
early_stop = EarlyStopping(monitor='val_loss',
                           patience=50,
                           restore_best_weights=True,
                           mode='min')

'''
Mediante las siguientes instrucciones se termina de cargar los datos necesarios para el entrenamiento

traingen = train_generator.flow_from_directory(train_data_dir,
                                               target_size=(224, 224),
                                               class_mode='categorical',
                                               subset='training',
                                               batch_size=BATCH_SIZE, 
                                               shuffle=True,
                                               seed=42)

validgen = train_generator.flow_from_directory(train_data_dir,
                                               target_size=(224, 224),
                                               class_mode='categorical',
                                               subset='validation',
                                               batch_size=BATCH_SIZE,
                                               shuffle=True,
                                               seed=42)

testgen = test_generator.flow_from_directory(test_data_dir,
                                             target_size=(224, 224),
                                             class_mode=None,
                                             batch_size=1,
                                             shuffle=False,
                                             seed=42)
'''

def createAutoEncoder(input_shape):
    """
    Creacion de un autoencoder sin compilar.

    Args:
        input_shape: tuple - tamaño de las imagenes (width, height, channels)
    Returns:
        encoder_input: Input - tensor que define el tamaño y forma de los datos que usara el modelo
        autoendoer: modelo de aprendizaje no supervisado autoencoder
    """
    # Definir la rama del Autoencoder
    encoder_input = Input(shape=input_shape)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(encoder_input)
    x = MaxPooling2D((2, 2), padding='same')(x)
    encoded = Conv2D(16, (3, 3), activation='relu', padding='same')(x)

    x = Conv2D(16, (3, 3), activation='relu', padding='same')(encoded)
    x = UpSampling2D((2, 2))(x)
    decoded = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)

    autoencoder = Model(encoder_input, decoded)

    return encoder_input, autoencoder

def createResNet50V2(input_shape):
    """
    Creacion de un modelo ResNet50V2. No se incluyen las capas totalmente conectadas de la capa superior y todas las capas se mantendan congeladas

    Args:
        input_shape: tuple - tamaño de las imagenes (width, height, channels)
    Returns:
        input_tensor: Input - tensor que define el tamaño y la forma de los datos que usara el modelo
        base_resnet: modelo ResNet50V2
    """
    input_tensor = Input(shape=input_shape)

    # Capas convolucionales preentrenadas son cargadas usando los pesos de Imagenet.
    # include_top es puesto a falso para excluirlas las capas totalmente conectadas en la parte superior
    base_resnet = ResNet50V2(include_top=False, weights='imagenet', input_shape=input_shape)

    return input_tensor, base_resnet


def create_model(input_shape, n_classes, optimizer='adam'):
    """
    Crea un modelo concatenado que usara una red ResNetV2 y un autoencoder para extraer caracteristicas y pasarlas a un clasificador.
    Este es un modelo concatenado que se usara para mejorar la eficiencia y rendimiento del programa.
    
    input_shape: tuple - tamaño de las imagenes (width, height, channels)
    n_classes: int - numero de clases de la capa de salida
    optimizer: string - Optimizador instanciado para usarloen el entrenamiento. Por defecto 'adam'
    """
    
    input_tensor, base_resnet = createResNet50V2(input_shape)
    #Definicion de la rama ResNet50V2
    resnet_branch = GlobalAveragePooling2D()(base_resnet(input_tensor))

    #Definicion de la rama AutoEncoder
    encoder_input, autoencoder_branch = createAutoEncoder(input_shape)
    autoencoder_branch.compile(optimizer=optimizer, loss='binary_crossentropy')

    # Aplana el resultado de la capa convolucional para que tanto la rama ResNet50V2 como la rama AutoEncoder tengan dimension (None, 2048)
    flattened_autoencoder_output = Flatten()(autoencoder_branch.output)

    # Concatenar las características extraídas
    concatenated_features = Concatenate()([resnet_branch, flattened_autoencoder_output])

    # Capas de clasificación final
    dense_layer = Dense(256, activation='relu')(concatenated_features)
    output_layer = Dense(n_classes, activation='softmax')(dense_layer)

    # Crear el modelo concatenado
    model = Model(inputs=[input_tensor, encoder_input], outputs=output_layer)

    # Compilar el modelo
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    
    
    return model


"""
algoritmo deteccion_en_tiempo_real:
    Acceder al servidor (Puede usarse tecnologías como paramiko para el acceso mediante SSH y el protocolo SFTP)
    Descargar ultima imagen del servidor
    Usar modelo de prediccion de mascaras para recortar la zona de agua

    predecir:
        Si es un vertido, avisar
        Terminar en caso contrario

Este algoritmo será neceario ejecutarlo cada vez que se introduzca una imagen en el servidor, el cual se ha establecido en 30 segundos.
Para realizar esta tarea será necesario usar cron. Esto permitira ejecutar un script cada cierto tiempo.
"""


modelo = create_model(input_shape, n_classes, optim_1)
modelo.summary()

'''
Para entrenar el modelo puede seguirse varios desarrollos:
    1. Puede crearse el modelo concatenado, entrenarlo y almacenarlo
    2. Crear los dos modelos que conforman el modelo concatenado, entrenalos con los datos
    individualmente, almacenarlos en diferentes archivos .keras y posteriormete concatenar 
    las caracteristicas extraidas por cada uno (recomendable)
'''



