import numpy as np
from PIL import Image

# Leer el archivo
with open('rpg.txt', 'r') as file:
    data = file.read()

# Limpiar y formatear los datos
data = data.replace("(", "").replace(")", "").replace(" ", "").split("\n")
pixels = []
cont1 = 1
for i in data:
    cont = 1
    for n in i.split(","):
        if n != "":
            pixels.append(int(n))
            if cont == 97*3:
                break
            cont +=1
    if cont1 == 97:
        break
    cont1 += 1



# Especificar las dimensiones de la imagen (ancho y alto)
width = 97  # Ejemplo de ancho
height = 97  # Ejemplo de alto

# Verificar que el número de píxeles sea consistente con las dimensiones especificadas
# if len(pixels) != width * height * 3:
#     raise ValueError("La cantidad de píxeles no coincide con las dimensiones especificadas.")

# Convertir la lista de píxeles en una matriz
pixels_matrix = np.array(pixels).reshape((height, width, 3))

# Convertir la matriz a una imagen
image = Image.fromarray(pixels_matrix.astype('uint8'))

# Guardar la imagen
image.save('imagen_rpg.png')

# Mostrar la imagen
image.show()
