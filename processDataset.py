import os
import cv2
import numpy as np
from PIL import Image
import shutil

BASE_DIR = ''
IMG_DIR = BASE_DIR + 'imagenes/' #Directorio que contiene el dataset de imagenes de los pozos, ordenados por carpetas
MASK_DIR = BASE_DIR + 'mascaras/' #Directorio que contiene el dataset de mascaras, ordenados por carpetas
PREDICTION_DIR = BASE_DIR + 'predicciones/'#Directorio que contendra las mascaras predichas, ordenadas por carpetas
MASK_APPLIED_DIR = BASE_DIR + 'Mascara aplicada/'


def loadImages(directorio, conjuntoDestino):
    """
    Metodo encargado de cargar las imagenes de un directorio

    Args:
        directorio: Directorio del cual se extraeran las imagenes
        conjuntoDestino: Conjunto que almacenara las rutas de las imagenes del directorio
    Returns:
        bool: False si no se puede realizar la carga, True en caso contrario
    """
    if os.path.exists(directorio):
        for img in os.listdir(directorio):
            ruta_imagen = os.path.join(directorio, img)

            if(ruta_imagen.endswith(('.jpg', '.jpeg', '.png'))):
                conjuntoDestino.append(ruta_imagen)
        if len(conjuntoDestino) != 0: return True

    else: return False

#Metodo en desuso por el cambio en la estructura del dataset
def loadDataset(directorio, conjuntoDestino):
    """
    Metodo encargado de cargar todo el Dataset contenido en el directorio que se pasa por parametros. Recorre la lista de directorios 
    contenidos en el directorio pasado por parametros y obtiene, para cada uno de ellos, el conjunto de imagenes.

    Args:
        directorio: directorio que contiene el Dataset que se quiere cargar
        conjuntoDestino: conjunto que almacenara el Dataset
    

    """

    if os.path.exists(directorio):
        for dir in os.listdir(directorio):
            conjuntoAuxiliar = []
            ruta_directorio = os.path.join(directorio, dir)

            if loadImages(ruta_directorio, conjuntoAuxiliar):
                conjuntoDestino.extend(conjuntoAuxiliar)
        if len(conjuntoDestino) != 0: return True
            
    else: return False


def processImages(conjuntoImagenes, conjuntoDestino):
    """
    Metodo encargado de procesar las imagenes del conjunto pasado por parametros

    Args:
        conjuntoImagenes: Conjunto que contiene las rutas de las imagenes que se van a procesar
        conjuntoDestino: Conjunto en el que se almacenaran las imagenes procesadas
    """
    conjuntoAux = []
    i=0
    for ruta_img in conjuntoImagenes:

            imagen = cv2.cvtColor(cv2.imread(ruta_img), cv2.COLOR_BGR2RGB)
            imagen = cv2.resize(imagen, (256, 256)) / 255.
            conjuntoAux.append(imagen)
            print(f"Imagen {i} procesada")
            i=i+1


    conjuntoDestino.clear()
    conjuntoDestino.extend(conjuntoAux)


def clean_dataset(conjuntoImagenes):
    """
    Metodo encargado de limpiar el dataset eliminando las imagenes que no tienen un formato correcto
    o aquellas en las cuales hubo un error al tomarla.

    Args:
        conjuntoImagenes: Conjunto que contiene las rutas de las imagenes
    """
    for ruta_img in conjuntoImagenes:
        try:
            with Image.open(ruta_img) as img:
                img.load()

                if img is None:
                    os.remove(ruta_img)
                    print(f"Imagen {ruta_img} eliminada")

        except Exception as e:
            os.remove(ruta_img)
            print(f"Imagen {ruta_img} eliminada")



def binarizarMascaras(conjuntoMascaras, conjuntoDestino):
    """
    Metodo encargado de binarizar las mascaras del Dataset

    Args:
        conjuntoMascaras: Conjunto que contiene las mascaras
        conjuntoDestino: Conjunto que almacenara las mascaras binarizadas
    """
    conjuntoAux = []
    i=0

    for masc in conjuntoMascaras:
        mascara = cv2.imread(masc)
        mascara = cv2.resize(mascara, (256, 256))
        mascara = 1.0 * (mascara[:, :, 0] > .1)

        mascara = np.expand_dims(mascara, axis=-1)

        conjuntoAux.append(mascara)
        print(f"Mascara {i} procesada")
        i=i+1

    conjuntoDestino.clear()
    conjuntoDestino.extend(conjuntoAux)



def almacenarMascaras(directorio, mascaras):
    """"
    Este metodo se encarga de almacenar mascaras en el directorio especificado

    Args:
        directorio: Directorio en el cual se almacenaran las mascaras
        mascaras: conjunto de mascaras que se quieren almacenar
    """
    if not os.path.exists(directorio):
        # Crea el directorio si no existe
        os.makedirs(directorio)

    for i, mask in enumerate(mascaras):
        # Escala la intensidad de la máscara a valores de 0 a 255 (uint8)
        mask_uint8 = (mask * 255).astype(np.uint8)

        mask_resized = cv2.resize(mask_uint8, (1920, 1080))

        # Almacena la máscara en escala de grises
        nombre_archivo = f"mask_{i}.png"
        ruta_archivo = os.path.join(directorio, nombre_archivo)
        cv2.imwrite(ruta_archivo, mask_resized)


def applyMask(imagenes, mascaras, directorio):
    """
    Este metodo se encarga de procesar imagenes aplicandoles una mascara binaria y almacenando el 
    resultado de dicha aplicacion en un directorio.

    Args:
        imagenes: Conjunto que contiene las imagenes a las que se le aplicara la mascara
        mascaras: Conjunto de mascaras que se van a aplicar
        directorio: Direcotorio que almacenara el resultado de la operacion
    """
    if not os.path.exists(directorio):
        os.makedirs(directorio)

    resultado = []
    for img, mask in zip(imagenes, mascaras):

        mask_applied = cv2.bitwise_and(img, mask)
        resultado.append(mask_applied)

    for i, res in enumerate(resultado):
        nombre_archivo = f"resultado_{i}.jpg"
        ruta_archivo = os.path.join(directorio, nombre_archivo)
        cv2.imwrite(ruta_archivo, res)



if __name__ == '__main__':
    conjDest = []
    #loadDataset(IMG_DIR, conjDest)
    #clean_dataset(conjDest)
    

