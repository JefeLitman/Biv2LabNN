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

    def __init__(self,directory_path, batch_size, shuffle = False):
        """Constructor de la clase UCF101.
        Args:
            directory_path: String que tiene la ruta absoluta de la ubicacion del
            dataset, incluyendo en el path el split.
            batch_size: Numero que corresponde al tamaño de elementos por batch
            shuffle: Booleano que determina si deben ser mezclados aleatoreamente los datos. Por defecto en False
            Aclaratoria es que esta clase incluye por defecto las carpetas de train, test y dev"""

        self.ds_directory = directory_path
        self.batch_size = batch_size
        self.batch_index = 0
        directories = os.listdir(self.ds_directory)

        for i in directories:
            if i.lower() == "train":
                self.train_path = os.path.join(self.ds_directory,i)
                self.dev_path = None
            elif i.lower() == "test":
                self.test_path = os.path.join(self.ds_directory,i)
                self.dev_path = None
            elif i.lower() == "dev":
                self.dev_path = os.path.join(self.ds_directory,i)
            else:
                raise ValueError(
                    'La organizacion de la carpeta debe seguir la estructura'
                    'de train, test y dev (este ultimo opcional) teniendo'
                    'los mismos nombres sin importar mayus o minus.'
                    'Carpeta del error: %s' % i)

        self. generar_clases()
        self.generar_videos_paths()

        if shuffle:
            self.shuffle_videos()

    def generar_clases(self):
        """Metodo que se encarga de generar el numero de clases, los nombres
        de clases, numeros con indices de clase y el diccionario que convierte de clase
        a numero como de numero a clase"""

        self.to_class = sorted(os.listdir(self.train_path)) #Equivale al vector de clases
        self.to_number = dict((name, index) for index,name in enumerate(self.to_class))

    def generar_videos_paths(self):
        """Metodo que se encarga de generar una lista con el path absoluto de todos los videos para train, test y
        dev si llega a aplicar esta carpeta.
        Args:
            dev_path: Booleano que me determina si existe el path de dev o de lo contario no existe.
            """
        self.videos_train_path = []
        self.videos_test_path = []
        self.videos_dev_path = []

        for clase in self.to_class:

            videos_train_path = os.path.join(self.train_path,clase)
            self.videos_train_path += [os.path.join(videos_train_path,i) for i in sorted(os.listdir(videos_train_path))]
            self.train_batches = int( len(self.videos_train_path) / self.batch_size)
            self.train_batches = self.train_batches + 1 if len(self.videos_train_path) % self.batch_size != 0 else self.train_batches

            videos_test_path = os.path.join(self.test_path,clase)
            self.videos_test_path += [os.path.join(videos_test_path,i) for i in sorted(os.listdir(videos_test_path))]
            self.test_batches = int( len(self.videos_test_path) /  self.batch_size)
            self.test_batches = self.test_batches + 1 if len(self.videos_test_path) % self.batch_size != 0 else self.test_batches

            if self.dev_path:
                videos_dev_path = os.path.join(self.dev_path,clase)
                self.videos_dev_path += [os.path.join(videos_dev_path,i) for i in sorted(os.listdir(videos_dev_path))]
                self.dev_batches = int( len(self.videos_dev_path) / self.batch_size)
                self.dev_batches = self.dev_batches + 1 if len(self.videos_dev_path) % self.batch_size != 0 else self.dev_batches

    def shuffle_videos(self):
        """Metodo que se encarga de realizar shuffle a los datos si esta
        activada la opcion de shuffle."""
        self.videos_train_path = np.random.permutation(self.videos_train_path)
        self.videos_test_path = np.random.permutation(self.videos_test_path)

        if self.dev_path:
            self.videos_dev_path = np.random.permutation(self.videos_dev_path)

    def get_frames_video(self,video_path, size = None, n_frames = None):
        """Metodo que se encarga de cargar en memoria los frames de un video a partir del path
        Args:
            video_path: String de la carpeta que contiene los frames del video.
            size: Tupla de la forma [new_height, new_widht]. Por defecto None donde no se redimensiona
            n_frames: Numero que corresponde a cuantos frames del video se van a tomar. Por
            defecto en None que corresponde a todos.
            """

        video = []
        frames = sorted(os.listdir(video_path))
        if(n_frames):
            crop_point = np.random.randint(0, len(frames) - n_frames)
            frames = frames[crop_point : crop_point + n_frames]

        for frame in frames:
            frame_raw = tf.io.read_file(os.path.join(video_path,frame))
            frame_tensor = tf.image.decode_image(frame_raw, channels=3,dtype=tf.float32)
            if size:
                frame_tensor = tf.image.resize_images(frame_tensor, size)
            video.append(frame_tensor)

        return tf.convert_to_tensor(video)

    def get_next_train_batch(self,image_size=None, n_frames=None):
        """Metodo que se encarga de retornar el siguiente batch o primer batch
        de datos cuando se es llamado.
        Args:
            image_size: Tupla de la forma [new_height, new_widht]. Por defecto None donde no se redimensiona.
            n_frames: Numero que corresponde a cuantos frames del video se van a tomar. Por
            defecto en None que corresponde a todos.
            """

        if self.batch_index > self.train_batches:
            self.batch_index = 0

        start_index = self.batch_index*self.batch_size
        if (self.batch_index + 1)*self.batch_size >= len(self.videos_train_path):
            end_index = len(self.videos_train_path)
        else:
            end_index = (self.batch_index + 1)*self.batch_size

        batch = []
        for index in range(start_index,end_index):
            video = self.get_frames_video(self.videos_train_path[index],size=image_size,n_frames=n_frames)
            batch.append(video)

        self.batch_index += 1

        return tf.convert_to_tensor(batch)
