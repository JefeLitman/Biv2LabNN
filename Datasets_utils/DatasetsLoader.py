"""UCF & HMDB dataset input module.
"""

import os
import tensorflow as tf
import warnings
import numpy as np

class VideoDataGenerator():
    """Clase para cargar todos los datos de un Dataset a partir de la ruta
    especificada por el usuario y ademas, agregando transformaciones en
    tiempo real.

    En esta version lee los datos a partir de los frames pero no archivos avi.
    """

    def __init__(self,directory_path, batch_size, shuffle = False):
        """Constructor de la clase.
        Args:
            directory_path: String que tiene la ruta absoluta de la ubicacion del
            dataset, incluyendo en el path el split.
            batch_size: Numero que corresponde al tamaño de elementos por batch
            shuffle: Booleano que determina si deben ser mezclados aleatoreamente los datos. Por defecto en False
            Aclaratoria es que esta clase incluye por defecto las carpetas de train, test y dev"""

        self.ds_directory = directory_path
        self.batch_size = batch_size
        directories = os.listdir(self.ds_directory)

        for i in directories:
            if i.lower() == "train":
                self.train_path = os.path.join(self.ds_directory,i)
                self.train_batch_index = 0
                self.dev_path = None
            elif i.lower() == "test":
                self.test_path = os.path.join(self.ds_directory,i)
                self.test_batch_index = 0
                self.dev_path = None
            elif i.lower() == "dev":
                self.dev_path = os.path.join(self.ds_directory,i)
                self.dev_batch_index = 0
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
                self.videos_dev_path = []
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

    def get_frames_video(self,video_path, size = None, n_frames = None, canales = 3):
        """Metodo que se encarga de cargar en memoria los frames de un video a partir del path
        Args:
            video_path: String de la carpeta que contiene los frames del video.
            size: Tupla de la forma [new_height, new_widht]. Por defecto None donde no se redimensiona
            n_frames: Numero que corresponde a cuantos frames del video se van a tomar. Por
            defecto en None que corresponde a todos.
            canales: Numero de canales que tienen las imagenes en las carpetas. Debe ser uniforme.
            """

        video = []
        frames = sorted(os.listdir(video_path))
        if(n_frames):
            crop_point = np.random.randint(0, len(frames) - n_frames)
            frames = frames[crop_point : crop_point + n_frames]

        for frame in frames:
            frame_path = os.path.join(video_path,frame)
            if frame_path.endswith(".jpeg") or frame_path.endswith(".jpg"):
                frame_tensor = tf.io.decode_jpeg(frame_path, channels=canales)
            elif frame_path.endswith(".png"):
                frame_tensor = tf.io.decode_png(frame_path, channels=canales)
            elif frame_path.endswith(".bmp"):
                frame_tensor = tf.io.decode_bmp(frame_path, channels=canales)
            else:
                raise ValueError(
                    'Los formatos validos de imagen para el generador son png, '
                    'jpeg, jpg o bmp. '
                    'Imagen del error: %s' % frame_path)
            frame_tensor = tf.image.convert_image_dtype(frame_tensor, dtype=tf.float32)
            if size:
                frame_tensor = tf.image.resize(frame_tensor, size)
            video.append(frame_tensor)

        return tf.convert_to_tensor(video)

    def get_next_train_batch(self,image_size=None, n_frames=None, n_canales = 3):
        """Metodo que se encarga de retornar el siguiente batch o primer batch
        de datos train cuando se es llamado.
        Args:
            image_size: Tupla de la forma [new_height, new_widht]. Por defecto None donde no se redimensiona.
            n_frames: Numero que corresponde a cuantos frames del video se van a tomar. Por
            defecto en None que corresponde a todos.
            n_canales: Numero que corresponde al numero de canales que posee las imagenes. Por defecto en 3.
            """

        if self.train_batch_index > self.train_batches:
            self.train_batch_index = 0

        start_index = self.train_batch_index*self.batch_size
        if (self.train_batch_index + 1)*self.batch_size >= len(self.videos_train_path):
            end_index = len(self.videos_train_path)
        else:
            end_index = (self.train_batch_index + 1)*self.batch_size

        batch = []
        labels = []
        for index in range(start_index,end_index):
            label = self.videos_train_path[index].split("/")[-2]
            video = self.get_frames_video(self.videos_train_path[index],size=image_size,n_frames=n_frames, canales=n_canales)
            labels.append(self.to_number[label])
            batch.append(video)

        self.train_batch_index += 1

        return tf.convert_to_tensor(batch), tf.convert_to_tensor(labels)

    def get_next_test_batch(self, image_size=None, n_frames=None, n_canales=3):
        """Metodo que se encarga de retornar el siguiente batch o primer batch
                de datos de test cuando se es llamado.
                Args:
                    image_size: Tupla de la forma [new_height, new_widht]. Por defecto None donde no se redimensiona.
                    n_frames: Numero que corresponde a cuantos frames del video se van a tomar. Por
                    defecto en None que corresponde a todos.
                    n_canales: Numero que corresponde al numero de canales que posee las imagenes. Por defecto en 3.
                    """

        if self.test_batch_index > self.test_batches:
            self.test_batch_index = 0

        start_index = self.test_batch_index * self.batch_size
        if (self.test_batch_index + 1) * self.batch_size >= len(self.videos_test_path):
            end_index = len(self.videos_test_path)
        else:
            end_index = (self.test_batch_index + 1) * self.batch_size

        batch = []
        labels = []
        for index in range(start_index, end_index):
            label = self.videos_train_path[index].split("/")[-2]
            video = self.get_frames_video(self.videos_train_path[index], size=image_size, n_frames=n_frames,
                                          canales=n_canales)
            labels.append(self.to_number[label])
            batch.append(video)

        self.test_batch_index += 1

        return tf.convert_to_tensor(batch), tf.convert_to_tensor(labels)