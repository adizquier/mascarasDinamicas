import os
os.environ["SM_FRAMEWORK"] = "tf.keras"
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
import shutil
import albumentations as A
import segmentation_models as sm
import keras.backend as K
sm.set_framework('tf.keras')
from skimage.color import rgb2gray
from skimage.transform import rotate
from deskew import determine_skew
from sklearn.model_selection import train_test_split
from keras.models import load_model
from tensorflow.keras import regularizers
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dropout

import processDataset as pd

#import torch, torchvision

sm.framework()

ALPHA = 0.8
GAMMA = 2

def DiceLoss(targets, inputs, smooth=1e-6):
    '''
    Funcion de perdida calculada segun el coeficiente de Dice
    '''
    
    #flatten label and prediction tensors
    inputs = tf.keras.backend.flatten(inputs)
    targets = tf.keras.backend.flatten(targets)
    
    intersection = K.sum(K.dot(targets, inputs))
    dice = (2*intersection + smooth) / (K.sum(targets) + K.sum(inputs) + smooth)
    return 1 - dice

def FocalLoss(targets, inputs, alpha=ALPHA, gamma=GAMMA):
    '''
    Funcion para el calculo de la perdida usando la funcion Focal Loss
    '''
    
    inputs = tf.keras.backend.flatten(inputs)
    targets = tf.keras.backend.flatten(targets)
    
    BCE = tf.keras.losses.binary_crossentropy(targets, inputs)
    BCE_EXP = tf.keras.backend.exp(-BCE)
    focal_loss = tf.keras.backend.mean(alpha * tf.keras.backend.pow((1-BCE_EXP), gamma) * BCE)
    
    return focal_loss

def make_image_gen(X_train, y_train, aug, batch_size):
    '''
    Funcion generadora para aumentar el conjunto de datos de entrenamiento
    '''
    aug_x = []
    aug_y = []
    while True:
        for i in range(X_train.shape[0]): 
            augmented = aug(image=X_train[i], mask=y_train[i])
            x, y = augmented['image'],  augmented['mask']
            aug_x.append(x)
            aug_y.append(y)
            if len(aug_x)>=batch_size:
                yield np.array(aug_x, dtype = 'float32'), np.array(aug_y, dtype = 'float32')
                aug_x, aug_y=[], []

def dice_coef(y_true, y_pred, smooth=1):
    '''
    Funcion para el calculo del coeficiente de Dice
    '''
    intersection = tf.keras.backend.sum(y_true * y_pred, axis=[1,2,3])
    union = tf.keras.backend.sum(y_true, axis=[1,2,3]) + tf.keras.backend.sum(y_pred, axis=[1,2,3])
    dice = tf.keras.backend.mean((2. * intersection + smooth)/(union + smooth), axis=0)
    return dice

#Objeto usado para la generacion de datos usando transformaciones
aug = A.Compose([
    A.OneOf([
        A.RandomSizedCrop(min_max_height=(50, 101), height=256, width=256, p=0.5),
        A.PadIfNeeded(min_height=256, min_width=256, p=0.5)
    ],p=1),
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
    A.OneOf([
        A.ElasticTransform(p=0.5, alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03),
        A.GridDistortion(p=0.5),
        A.OpticalDistortion(distort_limit=1, shift_limit=0.5, p=1),
    ], p=0.8)])

callbacks = [
    #tf.keras.callbacks.EarlyStopping(monitor="val_loss", mode="min", patience=15),
    tf.keras.callbacks.ModelCheckpoint(filepath='modelo_keras321.keras', monitor = 'val_loss', verbose = 1, save_best_only = False, mode = 'min'),
    #tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                                   #patience=5,
                                   #verbose=1, mode='min', min_delta=0.0001, cooldown=2, min_lr=1e-6)
]

def compileModel(classes=1, input_shape=(256,256,3), loss=FocalLoss, metrics=dice_coef, dropout_rate=0.5, l1_reg=0.01, l2_reg=0.01):
    """
    Metodo encargado de crear y compilar un modelo de red neuronal convolucional CNN

    Args:
        classes: Numero de clases diferentes que se deben segmentar
        input_shape: Tamaño de los datos de entrada
        loss: Funcion de perdida que usa el modelo
        metrics: Metrica usada por el modelo
    """
    base_model = sm.Unet('efficientnetb0', classes=classes, input_shape=input_shape, activation='sigmoid', encoder_weights='imagenet')

    # Añadir regularización L1/L2 a las capas densas
    for layer in base_model.layers:
        if isinstance(layer, tf.keras.layers.Dense):
            layer.kernel_regularizer = regularizers.l1_l2(l1=l1_reg, l2=l2_reg)
    
    # Obtener la salida del modelo base
    outputs = base_model.output

    # Agregar capa de dropout
    outputs = Dropout(dropout_rate)(outputs)

    # Crear el modelo completo
    model = Model(inputs=base_model.input, outputs=outputs)

    model.compile(optimizer=tf.keras.optimizers.Adam(0.001), loss=loss, metrics=[metrics])

    return model

def trainModel(model, X_train, y_train, aug, step_per_epoch, epochs, callback, X_test, y_test):
    """
    Metodo encargado de entrenarel un modelo de red neuronal

        y_train: Conjunto de etiquetas para el entrenamiento
        aug: Objeto usado en la generacion de datos usando transformaciones
        epochs: Numero de epocas
        callback: Funcion de callback
        X_test: Conjunto de datos de test
        y_test: COnjunto de etiquetas de test
    """
    model.fit(x=make_image_gen(X_train, y_train, aug, 16), steps_per_epoch = step_per_epoch, epochs=epochs, callbacks = callback, validation_data = (X_test, y_test))


def train_model_dataset():
    """
    Metodo que se encarga de entrenar un modelo de red neuronal para realizar la prediccion de mascaras dinamicas. El algoritmo se ha
    desarrollado para un estructura especifica de dataset, la cual consiste en la division del mismo por EDAR, cada una de ellas divididas
    en diferentes niveles segun las necesidades del problema.

    Se realiza un entrenamiento por lotes mixto, por lo que para cada EDAR se entrenaran los diferentes niveles a la vez. Como el número de
    elementos totales en este tipo de entrenamiento es tan grande que necesita de muchos recursos, se ha dividido el dataset con un factor de
    división. Un valor de 2 divide el dataset un 50% mientras que un valor de 4 lo divide un 25%.
    """
    
    model = compileModel()
    factor_division = 2

    for edar_imgs, edar_masks in zip (sorted(os.listdir(pd.IMG_DIR)), sorted(os.listdir(pd.MASK_DIR))):
        edar_imgs_path = os.path.join(pd.IMG_DIR, edar_imgs)
        edar_masks_path = os.path.join(pd.MASK_DIR, edar_masks)

        for i in range(factor_division):

            all_images = []
            all_masks = []

            for imagenes_level, mascaras_level in zip(sorted(os.listdir(edar_imgs_path)), sorted(os.listdir(edar_masks_path))):
                level_path = os.path.join(edar_imgs_path, imagenes_level)
                mascaras_level_path = os.path.join(edar_masks_path, mascaras_level)

                imagenes = []
                mascaras = []

                out_rgb = []
                out_masks = []

                pd.loadImages(level_path, imagenes)
                pd.loadImages(mascaras_level_path, mascaras)
                
                x = i*len(imagenes)//factor_division
                y = (i+1)*len(imagenes)//factor_division
                
                pd.processImages(imagenes[x : y], out_rgb)
                pd.binarizarMascaras(mascaras[x : y], out_masks)

                all_images.extend(out_rgb)
                all_masks.extend(out_masks)
            
            #El único motivo es liberar memoria antes de entrenar, si no se quedan los ultimos datos almacenados
            out_rgb.clear()
            out_masks.clear()
            imagenes.clear()
            mascaras.clear()

            all_images = np.array(all_images, dtype='float32')
            all_masks = np.array(all_masks, dtype='float32')

            print(f"Total de imágenes: {len(all_images)}\t Total de máscaras: {len(all_masks)}")

            # Dividir el conjunto de datos en conjunto de entrenamiento y conjunto de prueba
            X_train, X_test, y_train, y_test = train_test_split(all_images, all_masks, test_size=0.5, shuffle=True)

            trainModel(model, X_train, y_train, aug, 200, 50, callbacks, X_test, y_test)


        
            

#Modificar para incluir la nueva estructura de Dataset.
def realizarPredicciones(modelo):
    """
    El objetivo de este metodo consiste en realizar un recorrido de todas las imagenes del dataset, realizar las predicciones oportunas por un modelo
    de red neuronal de segmentacion, y almacenar las predicciones organizandolas por la misma estructura que el dataset de imagenes

    Args:
        modelo: modelo de red, previamente compilado y entrenado, que se utilizara para realizar las predicciones
    """

    #Si no existe el directorio, lo crea
    if not os.path.exists(pd.PREDICTION_DIR):
        os.mkdir(pd.PREDICTION_DIR)
    
    #Se recorre el conjunto de directorios del dataset de imagenes, organizado por cada EDAR
    for dir in os.listdir(pd.IMG_DIR):
        imagenes = []
        directorio = os.path.join(pd.IMG_DIR, dir)

        #Se cargan y procesan las imagenes del directorio
        pd.loadImages(directorio, imagenes)
        pd.processImages(imagenes, imagenes)

        imagenes = np.array(imagenes, dtype = 'float32')
        pred_dir = os.path.join(pd.PREDICTION_DIR, dir)
        
        #Se crea, si no existe, el directorio que contendrá las predicciones para una EDAR
        if not os.path.exists(pred_dir):
            os.mkdir(pred_dir)
            preds = modelo.predict(imagenes)
            pd.almacenarMascaras(pred_dir, preds)
        #Si existe, se comprueba el numero de elementos del directorio. Si es igual al numero de elementos del conjunto de imagenes, no se hace nada
        #Si es distinto, se eliminan las predicciones previas y se vuelve a predecir, pues el conjunto de imagenes de la EDAR han aumentado
        else:
            if len(imagenes) != len(pred_dir):
                shutil.rmtree(pred_dir)
                pred = modelo.predict(imagenes)
                pd.almacenarMascaras(pred_dir, pred)

def predecir(modelo, directorio, directorio_destino):
    """
    Este metodo se encarga de usar el modelo pasado por parametros y realizar predicciones de las imagenes contenidas en el direcorio que se pasa por parametros.
    El resultado de las predicciones se almacenan en un conjunto para su posterior almacenamiento.

    Args:
        modelo: Modelo de red que se utilizara para las predicciones
        directorio: Directorio que contiene las imagenes para las que se va a realizar las predicciones
        directorio_destino: Directorio en el cual se almacenaran las predicciones
    """
    if not os.path.exists(directorio_destino):
        os.mkdir(directorio_destino)
    
    imagenes = []
    if(os.path.exists(directorio)):
        pd.loadImages(directorio, imagenes)
        pd.processImages(imagenes, imagenes)
        
        imagenes = np.array(imagenes, dtype = 'float32')

        preds = modelo.predict(imagenes)
        pd.almacenarMascaras(directorio_destino, preds)




if __name__ == '__main__':
    train_model_dataset()