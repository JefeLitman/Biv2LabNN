"""UCF & HMDB dataset input module.
"""

import os
import tensorflow as tf
import warnings
import numpy as np

class UCF101():
    """Objeto para cargar todos los datos del Dataset ucf101
    en forma secuencial entregando batch por batch en entrenamiento
    o la totalidad de los datos en test.

    En esta version lee los datos a partir de los frames pero no archivos avi.

    Futuramente seria bueno generar un VideoDataGenerator para cualquier
    tipo de dataset."""

    def __init__(self,directory_path, batch_size):
        """Constructor de la clase UCF101.
        Args:
            directory_path: String que tiene la ruta absoluta de la ubicacion del
            dataset, incluyendo en el path el split.
            batch_size: Tamaño de elementos por batch
            Aclaratoria es que esta clase incluye por defecto las carpetas de train, test y dev"""

        self.ds_directory = directory_path
        self.batch_size = batch_size
        directories = os.listdir(self.ds_directory)

        for i in directories:
            if i.lower() == "train":
                self.train_path = os.path.join(self.ds_directory,i)
            elif i.lower() == "test":
                self.test_path = os.path.join(self.ds_directory,i)
            elif i.lower() == "dev":
                self.dev_path = os.path.join(self.ds_directory,i)
            else:
                raise ValueError(
                    'La organizacion de la carpeta debe seguir la estructura'
                    'de train, test y dev (este ultimo opcional) teniendo'
                    'los mismos nombres sin importar mayus o minus.'
                    'Carpeta del error: %s' % i)

        self. generar_clases()

    def generar_clases(self):
        """Metodo que se encarga de generar el numero de clases, los nombres
        de clases, numeros con indices de clase y el diccionario que convierte de clase
        a numero como de numero a clase"""

        self.to_class = sorted(os.listdir(self.train_path))
        self.to_number = dict((name, index) for index,name in enumerate(self.to_class))

    def generar_videos_paths(self):
        self.videos_path = []
        for clase in self.to_class:
            