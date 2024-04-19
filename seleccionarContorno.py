import cv2
import os

def mostrarImagen(titulo, img, Size=(1000, 1000)):
    cv2.namedWindow(titulo, cv2.WINDOW_NORMAL)  # Permite redimensionar la ventana
    cv2.resizeWindow(titulo, Size[0], Size[1])  # Establece el tamaño de la ventana
    cv2.imshow(titulo, img)

ruta_mascaras = './Mascara aplicada/Madrigalejo'

for img in os.listdir(ruta_mascaras):
    ruta = os.path.join(ruta_mascaras, img)

    imagen = cv2.imread(ruta)

    gray = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    blurred_image = cv2.GaussianBlur(gray, (35,35), 0)

    _,thres = cv2.threshold(blurred_image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    c, _ = cv2.findContours(thres, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

    contour_image = imagen.copy()
    cv2.drawContours(contour_image, c, -1, (0,255,0), 2)
    mostrarImagen('Imagen', contour_image)
    cv2.waitKey(0)