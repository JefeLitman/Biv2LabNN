"""UCF & HMDB dataset input module.
"""

import os
import cv2
import warnings
import numpy as np

class VideoDataGenerator():
    """Clase para cargar todos los datos de un Dataset a partir de la ruta
    especificada por el usuario y ademas, agregando transformaciones en
    tiempo real.

    En esta version lee los datos a partir de los frames pero no archivos avi.
    """

    def __init__(self,directory_path,
                 batch_size = 32,
                 frame_size=(None,None),
                 video_frames = 16,
                 temporal_crop = (None, None),
                 frame_crop = (None, None),
                 frame_transformation = (None, None),
                 video_rotation = (None,  None),
                 shuffle = False
                 ):
        """Constructor de la clase.
        Args:
            directory_path: String que tiene la ruta absoluta de la ubicacion del
            dataset, incluyendo en el path el split.
            batch_size: Numero que corresponde al tamaño de elementos por batch
            frame_size: Tupla de enteros de la estructura (width, height).
            video_frames: Numero de frames con los que quedara los batch
            shuffle: Booleano que determina si deben ser mezclados aleatoreamente los datos. Por defecto en False

            Aclaratoria es que esta clase incluye por defecto las carpetas de train, test y dev,
            ademas, siempre usa la notacion de canales al final"""

        """Definicion de constantes, atributos y restricciones a los parametros"""
        temporal_crop_modes = (None,'sequential','random','custom')
        video_frames_crop_modes = (None, )

        self.ds_directory = directory_path
        self.batch_size = batch_size
        self.frame_size = frame_size
        self.video_frames = video_frames
        self.transformation_index = 0

        """Proceso de revisar que los directorios del path esten en la jerarquia correcta"""
        directories = os.listdir(self.ds_directory)
        for i in directories:
            if i.lower() == "train":
                self.train_path = os.path.join(self.ds_directory,i)
                self.train_batch_index = 0
                self.train_data = []
                self.dev_path = None
                self.dev_data = None
            elif i.lower() == "test":
                self.test_path = os.path.join(self.ds_directory,i)
                self.test_batch_index = 0
                self.test_data = []
                self.dev_path = None
                self.dev_data = None
            elif i.lower() == "dev":
                self.dev_path = os.path.join(self.ds_directory,i)
                self.dev_batch_index = 0
                self.dev_data = []
            else:
                raise ValueError(
                    'La organizacion de la carpeta debe seguir la estructura'
                    'de train, test y dev (este ultimo opcional) teniendo'
                    'los mismos nombres sin importar mayus o minus.'
                    'Carpeta del error: %s' % i)

        self. generate_classes()
        self.generate_videos_paths()

    def generate_classes(self):
        """Metodo que se encarga de generar el numero de clases, los nombres
        de clases, numeros con indices de clase y el diccionario que convierte de clase
        a numero como de numero a clase"""

        self.to_class = sorted(os.listdir(self.train_path)) #Equivale al vector de clases
        self.to_number = dict((name, index) for index,name in enumerate(self.to_class))

    def generate_videos_paths(self):
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

            videos_test_path = os.path.join(self.test_path,clase)
            self.videos_test_path += [os.path.join(videos_test_path,i) for i in sorted(os.listdir(videos_test_path))]

            if self.dev_path:
                self.videos_dev_path = []
                videos_dev_path = os.path.join(self.dev_path,clase)
                self.videos_dev_path += [os.path.join(videos_dev_path,i) for i in sorted(os.listdir(videos_dev_path))]
                
        self.train_batches = int( len(self.videos_train_path) / self.batch_size)
        residuo = len(self.videos_train_path) % self.batch_size
        if residuo != 0:
            self.train_batches += 1
            self.videos_train_path += self.videos_train_path[:self.batch_size - residuo]
        
        self.test_batches = int( len(self.videos_test_path) /  self.batch_size)
        residuo = len(self.videos_test_path) % self.batch_size
        if residuo != 0:
            self.test_batches += 1
            self.videos_test_path += self.videos_test_path[:self.batch_size - residuo]
        
        if self.dev_path:
            self.dev_batches = int( len(self.videos_dev_path) / self.batch_size)
            residuo = len(self.videos_dev_path) % self.batch_size
            if residuo != 0:
                self.dev_batches += 1
                self.videos_dev_path += self.videos_dev_path[:self.batch_size - residuo]

    def shuffle_videos(self):
        """Metodo que se encarga de realizar shuffle a los datos si esta
        activada la opcion de shuffle."""
        self.videos_train_path = np.random.permutation(self.videos_train_path)
        self.videos_test_path = np.random.permutation(self.videos_test_path)

        if self.dev_path:
            self.videos_dev_path = np.random.permutation(self.videos_dev_path)

    def get_frames_video(self,video_path, size = None, n_frames = None, channels = 3):
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
            frame_tensor = cv2.imread(frame_path)
            if size:
                frame_tensor = cv2.resize(frame_tensor, tuple(size))
            video.append(frame_tensor)

        return np.asarray(video, dtype=np.float32)

    def get_next_train_batch(self,image_size=None, n_frames=None, n_canales = 3):
        """Metodo que se encarga de retornar el siguiente batch o primer batch
        de datos train cuando se es llamado.
        Args:
            image_size: Tupla de la forma [new_height, new_widht]. Por defecto None donde no se redimensiona.
            n_frames: Numero que corresponde a cuantos frames del video se van a tomar. Por
            defecto en None que corresponde a todos.
            n_canales: Numero que corresponde al numero de canales que posee las imagenes. Por defecto en 3.
            """

        if self.train_batch_index >= self.train_batches:
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
            video = self.get_frames_video(self.videos_train_path[index],size=image_size,n_frames=n_frames, channels=n_canales)
            labels.append(self.to_number[label])
            batch.append(video)

        self.train_batch_index += 1

        return np.asarray(batch, dtype=np.float32), np.asarray(labels, dtype=np.int64)

    def get_next_test_batch(self, image_size=None, n_frames=None, n_canales=3):
        """Metodo que se encarga de retornar el siguiente batch o primer batch
                de datos de test cuando se es llamado.
                Args:
                    image_size: Tupla de la forma [new_height, new_widht]. Por defecto None donde no se redimensiona.
                    n_frames: Numero que corresponde a cuantos frames del video se van a tomar. Por
                    defecto en None que corresponde a todos.
                    n_canales: Numero que corresponde al numero de canales que posee las imagenes. Por defecto en 3.
                    """

        if self.test_batch_index >= self.test_batches:
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
                                          channels=n_canales)
            labels.append(self.to_number[label])
            batch.append(video)

        self.test_batch_index += 1

        return np.asarray(batch, dtype=np.float32), np.asarray(labels, dtype=np.int64)

    def load_raw_frame(self,frame_path, channels = 3):
        """Metodo que se encarga de cargar los frames dada la ruta en memoria
        Args:
            frame_path: String que posee la ruta absoluta del frame
            channels: Entero opcional, que corresponde a cuantos canales desea cargar la imagen"""
        if channels == 1:
            return cv2.imread(frame_path, cv2.COLOR_BGR2GRAY)
        elif channels == 3:
            return cv2.imread(frame_path, cv2.COLOR_BGR2RGB)
        else:
            return cv2.imread(frame_path)

    def resize_frame(self, image):
        """Metodo que se encarga de redimensionar un frame segun el tamaño
        especificado por el usuario"""
        return cv2.resize(image, tuple(self.frame_size))

    def crop_frame(self, frame, x, y):
        return frame[x:x+self.frame_size[1], y:y+self.frame_size[0]]

    def temporal_crop(self, mode , custom_fn):
        """Metodo que se encarga de realizar el corte temporal en los videos de
        train, test y dev segun el modo especificado y los agrega a la lista de datos.
        Args:
            mode: String o None que corresponde al modo de aumento de datos.
            custom_fn: Callback o funcion de python que retorna la lista de los path a cargar,
            """
        if mode == 'sequential':
            """ Modo secuencial, donde se toman todos los frames del video en forma
            secuencial hasta donde el video lo permita"""
            for video in self.videos_train_path:
                frames_path = sorted(os.listdir(video))
                while len(frames_path) < self.video_frames:
                    frames_path += frames_path[:self.video_frames - len(frames_path)]
                label = self.to_number[video.split("/")[-2]]

                n_veces = len(frames_path) // self.video_frames
                for i in range(n_veces):
                    start = self.video_frames * i
                    end = self.video_frames * (i+1)
                    frames = frames_path[start:end]

                    name = "tcrop" + str(self.transformation_index)
                    elemento = { (name, None) : (frames, label) }
                    self.transformation_index += 1
                    self.train_data.append(elemento)

            for video in self.videos_test_path:
                frames_path = sorted(os.listdir(video))
                while len(frames_path) < self.video_frames:
                    frames_path += frames_path[:self.video_frames - len(frames_path)]
                label = self.to_number[video.split("/")[-2]]

                n_veces = len(frames_path) // self.video_frames
                for i in range(n_veces):
                    start = self.video_frames * i
                    end = self.video_frames * (i + 1)
                    frames = frames_path[start:end]

                    name = "tcrop" + str(self.transformation_index)
                    elemento = {(name, None): (frames, label)}
                    self.transformation_index += 1
                    self.test_data.append(elemento)

            if self.dev_path:
                for video in self.videos_dev_path:
                    frames_path = sorted(os.listdir(video))
                    while len(frames_path) < self.video_frames:
                        frames_path += frames_path[:self.video_frames - len(frames_path)]
                    label = self.to_number[video.split("/")[-2]]

                    n_veces = len(frames_path) // self.video_frames
                    for i in range(n_veces):
                        start = self.video_frames * i
                        end = self.video_frames * (i + 1)
                        frames = frames_path[start:end]

                        name = "tcrop" + str(self.transformation_index)
                        elemento = {(name, None): (frames, label)}
                        self.transformation_index += 1
                        self.dev_data.append(elemento)

        elif mode == 'random':
            """Modo aleatorio, donde la funcion personalizada corresponde al numero
             de cortes aleatorio por video que se vayan a hacer. Estos cortes mantienen
             la secuencia temporal pero puede tomar el inicio desde la cola y terminar en
             el inicio de los videos."""
            if isinstance(custom_fn, int):
                n_veces = custom_fn
            else:
                raise ValueError(
                    'Al usar el modo de corte temporal aleatorio, custom_fn debe ser un entero'
                    ', el valor entregado es de tipo: %s' % type(custom_fn)
                )
            for video in self.videos_train_path:
                frames_path = sorted(os.listdir(video))
                while len(frames_path) < self.video_frames:
                    frames_path += frames_path[:self.video_frames - len(frames_path)]
                label = self.to_number[video.split("/")[-2]]

                for _ in range(n_veces):
                    start = np.random.randint(0,len(frames_path))
                    if start + self.video_frames > len(frames_path):
                        end = self.video_frames + start - len(frames_path)
                        frames = frames_path[start : ] + frames_path[ : end]
                    else:
                        end = start + self.video_frames
                        frames = frames_path[start : end]

                    name = "tcrop" + str(self.transformation_index)
                    elemento = { (name, None) : (frames, label) }
                    self.transformation_index += 1
                    self.train_data.append(elemento)

            for video in self.videos_test_path:
                frames_path = sorted(os.listdir(video))
                while len(frames_path) < self.video_frames:
                    frames_path += frames_path[:self.video_frames - len(frames_path)]
                label = self.to_number[video.split("/")[-2]]

                for _ in range(n_veces):
                    start = np.random.randint(0, len(frames_path))
                    if start + self.video_frames > len(frames_path):
                        end = self.video_frames + start - len(frames_path)
                        frames = frames_path[start:] + frames_path[: end]
                    else:
                        end = start + self.video_frames
                        frames = frames_path[start: end]

                    name = "tcrop" + str(self.transformation_index)
                    elemento = {(name, None): (frames, label)}
                    self.transformation_index += 1
                    self.test_data.append(elemento)

            if self.dev_path:
                for video in self.videos_dev_path:
                    frames_path = sorted(os.listdir(video))
                    while len(frames_path) < self.video_frames:
                        frames_path += frames_path[:self.video_frames - len(frames_path)]
                    label = self.to_number[video.split("/")[-2]]

                    for _ in range(n_veces):
                        start = np.random.randint(0, len(frames_path))
                        if start + self.video_frames > len(frames_path):
                            end = self.video_frames + start - len(frames_path)
                            frames = frames_path[start:] + frames_path[: end]
                        else:
                            end = start + self.video_frames
                            frames = frames_path[start: end]

                        name = "tcrop" + str(self.transformation_index)
                        elemento = {(name, None): (frames, label)}
                        self.transformation_index += 1
                        self.dev_data.append(elemento)

        elif mode == 'custom':
            """Metodo que se encarga de ejecutar la funcion customizada a cada video
            y ejecutar el metodo para obtener los datos a agregar."""
            if custom_fn:
                for video in self.videos_train_path:
                    frames_path = sorted(os.listdir(video))
                    frames = custom_fn(frames_path)
                    label = self.to_number[video.split("/")[-2]]

                    try:
                        n_veces = len(frames)
                        for i in range(n_veces):
                            if len(frames[i]) != self.video_frames:
                                raise ValueError(
                                    'La longitud de los frames a agregar por medio de una '
                                    'funcion customizada debe ser igual a la especificada '
                                    'por el usuario. Longitud encontrada de %s' % len(frames[i])
                                )
                            name = "tcrop" + str(self.transformation_index)
                            elemento = {(name, None): (frames[i], label)}
                            self.transformation_index += 1
                            self.train_data.append(elemento)

                    except:
                        raise AttributeError(
                            'Se espera que la funcion customizada retorne una matriz'
                            ' donde cada fila corresponde a un video con el corte temporal '
                            'y la dimension de columnas sea igual a la longitud de frames'
                            ' especificada'
                        )

                for video in self.videos_test_path:
                    frames_path = sorted(os.listdir(video))
                    frames = custom_fn(frames_path)
                    label = self.to_number[video.split("/")[-2]]

                    try:
                        n_veces = len(frames)
                        for i in range(n_veces):
                            if len(frames[i]) != self.video_frames:
                                raise ValueError(
                                    'La longitud de los frames a agregar por medio de una '
                                    'funcion customizada debe ser igual a la especificada '
                                    'por el usuario. Longitud encontrada de %s' % len(frames[i])
                                )
                            name = "tcrop" + str(self.transformation_index)
                            elemento = {(name, None): (frames[i], label)}
                            self.transformation_index += 1
                            self.test_data.append(elemento)

                    except:
                        raise AttributeError(
                            'Se espera que la funcion customizada retorne una matriz'
                            ' donde cada fila corresponde a un video con el corte temporal '
                            'y la dimension de columnas sea igual a la longitud de frames'
                            ' especificada'
                        )

                if self.dev_path:
                    for video in self.videos_dev_path:
                        frames_path = sorted(os.listdir(video))
                        frames = custom_fn(frames_path)
                        label = self.to_number[video.split("/")[-2]]

                        try:
                            n_veces = len(frames)
                            for i in range(n_veces):
                                if len(frames[i]) != self.video_frames:
                                    raise ValueError(
                                        'La longitud de los frames a agregar por medio de una '
                                        'funcion customizada debe ser igual a la especificada '
                                        'por el usuario. Longitud encontrada de %s' % len(frames[i])
                                    )
                                name = "tcrop" + str(self.transformation_index)
                                elemento = {(name, None): (frames[i], label)}
                                self.transformation_index += 1
                                self.dev_data.append(elemento)

                        except:
                            raise AttributeError(
                                'Se espera que la funcion customizada retorne una matriz'
                                ' donde cada fila corresponde a un video con el corte temporal '
                                'y la dimension de columnas sea igual a la longitud de frames'
                                ' especificada'
                            )
            else:
                raise ValueError('Debe pasar la funcion customizada para el '
                    'modo customizado, de lo contrario no podra usarlo. Tipo de dato'
                     ' de funcion customizada recibida: %s' % type(custom_fn))

        else:
            """Modo None, donde se toman los primeros frames del video"""
            for video in self.videos_train_path:
                name = "tcrop" + str(self.transformation_index)
                frames_path = sorted(os.listdir(video))
                while len(frames_path) < self.video_frames:
                    frames_path += frames_path[:self.video_frames - len(frames_path)]
                frames_path = frames_path[:self.video_frames]
                label = self.to_number[video.split("/")[-2]]
                elemento = { (name, None) : (frames_path, label) }
                self.transformation_index += 1
                self.train_data.append(elemento)

            for video in self.videos_test_path:
                name = "tcrop" + str(self.transformation_index)
                frames_path = sorted(os.listdir(video))
                while len(frames_path) < self.video_frames:
                    frames_path += frames_path[:self.video_frames - len(frames_path)]
                frames_path = frames_path[:self.video_frames]
                label = self.to_number[video.split("/")[-2]]
                elemento = {(name, None): (frames_path, label)}
                self.transformation_index += 1
                self.test_data.append(elemento)

            if self.dev_path:
                for video in self.videos_dev_path:
                    name = "tcrop" + str(self.transformation_index)
                    frames_path = sorted(os.listdir(video))
                    while len(frames_path) < self.video_frames:
                        frames_path += frames_path[:self.video_frames - len(frames_path)]
                    frames_path = frames_path[:self.video_frames]
                    label = self.to_number[video.split("/")[-2]]
                    elemento = {(name, None): (frames_path, label)}
                    self.transformation_index += 1
                    self.dev_data.append(elemento)

    def frame_crop(self,mode, custom_fn, conserve_original = False):
        """Metodo que se encarga de realizar el corte de una imagen segun el
        tamaño especificado por el usuario siguiendo el modo elegido. Tambien
        sirve para aplicar transformaciones sobre los frames ya que esta operacion
        se hace uno por uno. Al final agregara las transforamciones a la lista
        de datos reemplazando o inmediatamente despues si se quieren conservar.
        Args:
            mode: String o None que corresponde al modo de aumento de datos.
            custom_fn: Callback o funcion de python que retorna la lista de los path a cargar
            converse_original: Booleano por defecto en False que indica si se agregan o reemplazan
            los valores ya almacenados en la lista de datos.
            """
        if mode == 'sequential':
            """Modo secuencial, donde se toman por cada imagen (desde izq a der)
             y arriba hacia abajo el tamaño indicado porel usuario hasta donde se
             le permita"""
            if conserve_original:
                n = len(self.train_data)
                for index in range(n):
                    #Agrego los nuevos cortes de frames a los datos
                    values = tuple(self.train_data[index].values())[0]
                    original_height = self.load_raw_frame(values[0][0]).shape[0]
                    original_width = self.load_raw_frame(values[0][0]).shape[1]
                    n_veces = min([original_width // self.frame_size[0], original_height//self.frame_size[1]])

                    for i in range(n_veces):
                        for j in range(n_veces):
                            start_width = i * self.frame_size[0]
                            end_width = start_width + self.frame_size[0]
                            start_height = j * self.frame_size[1]
                            end_height = start_height + self.frame_size[1]
                            function = lambda frame: frame[start_height : end_height, start_width : end_width]

                            name = "icrop" + str(self.transformation_index)
                            self.transformation_index += 1
                            elemento = { (name, function) : values }
                            self.train_data.append(elemento)

                    #Reemplazo la funcion de los datos ya almacenados
                    llave_original = tuple(self.train_data[index].keys())[0]
                    llave_nueva = (llave_original[0] + "icrop" + str(self.transformation_index), self.resize_frame)
                    valores = tuple(self.train_data[index].values())[0]
                    self.train_data[index] = {llave_nueva: valores}

                n = len(self.test_data)
                for index in range(n):
                    # Agrego los nuevos cortes de frames a los datos
                    values = tuple(self.test_data[index].values())[0]
                    original_height = self.load_raw_frame(values[0][0]).shape[0]
                    original_width = self.load_raw_frame(values[0][0]).shape[1]
                    n_veces = min([original_width // self.frame_size[0], original_height // self.frame_size[1]])

                    for i in range(n_veces):
                        for j in range(n_veces):
                            start_width = i * self.frame_size[0]
                            end_width = start_width + self.frame_size[0]
                            start_height = j * self.frame_size[1]
                            end_height = start_height + self.frame_size[1]
                            function = lambda frame: frame[start_height: end_height, start_width: end_width]

                            name = "icrop" + str(self.transformation_index)
                            self.transformation_index += 1
                            elemento = {(name, function): values}
                            self.test_data.append(elemento)

                    # Reemplazo la funcion de los datos ya almacenados
                    llave_original = tuple(self.test_data[index].keys())[0]
                    llave_nueva = (llave_original[0] + "icrop" + str(self.transformation_index), self.resize_frame)
                    valores = tuple(self.test_data[index].values())[0]
                    self.test_data[index] = {llave_nueva: valores}

                if self.dev_path:
                    n = len(self.dev_data)
                    for index in range(n):
                        # Agrego los nuevos cortes de frames a los datos
                        values = tuple(self.dev_data[index].values())[0]
                        original_height = self.load_raw_frame(values[0][0]).shape[0]
                        original_width = self.load_raw_frame(values[0][0]).shape[1]
                        n_veces = min([original_width // self.frame_size[0], original_height // self.frame_size[1]])

                        for i in range(n_veces):
                            for j in range(n_veces):
                                start_width = i * self.frame_size[0]
                                end_width = start_width + self.frame_size[0]
                                start_height = j * self.frame_size[1]
                                end_height = start_height + self.frame_size[1]
                                function = lambda frame: frame[start_height: end_height, start_width: end_width]

                                name = "icrop" + str(self.transformation_index)
                                self.transformation_index += 1
                                elemento = {(name, function): values}
                                self.dev_data.append(elemento)

                        # Reemplazo la funcion de los datos ya almacenados
                        llave_original = tuple(self.dev_data[index].keys())[0]
                        llave_nueva = (llave_original[0] + "icrop" + str(self.transformation_index), self.resize_frame)
                        valores = tuple(self.dev_data[index].values())[0]
                        self.dev_data[index] = {llave_nueva: valores}
            else:
                n = len(self.train_data)
                new_train_data = []
                for index in range(n):
                    # Agrego los nuevos cortes de frames a los nuevos datos
                    values = tuple(self.train_data[index].values())[0]
                    original_height = self.load_raw_frame(values[0][0]).shape[0]
                    original_width = self.load_raw_frame(values[0][0]).shape[1]
                    n_veces = min([original_width // self.frame_size[0], original_height // self.frame_size[1]])

                    for i in range(n_veces):
                        for j in range(n_veces):
                            start_width = i * self.frame_size[0]
                            end_width = start_width + self.frame_size[0]
                            start_height = j * self.frame_size[1]
                            end_height = start_height + self.frame_size[1]
                            function = lambda frame: frame[start_height: end_height, start_width: end_width]

                            name = "icrop" + str(self.transformation_index)
                            self.transformation_index += 1
                            elemento = {(name, function): values}
                            new_train_data.append(elemento)
                self.train_data = new_train_data

                n = len(self.test_data)
                new_test_data = []
                for index in range(n):
                    # Agrego los nuevos cortes de frames a los nuevos datos
                    values = tuple(self.test_data[index].values())[0]
                    original_height = self.load_raw_frame(values[0][0]).shape[0]
                    original_width = self.load_raw_frame(values[0][0]).shape[1]
                    n_veces = min([original_width // self.frame_size[0], original_height // self.frame_size[1]])

                    for i in range(n_veces):
                        for j in range(n_veces):
                            start_width = i * self.frame_size[0]
                            end_width = start_width + self.frame_size[0]
                            start_height = j * self.frame_size[1]
                            end_height = start_height + self.frame_size[1]
                            function = lambda frame: frame[start_height: end_height, start_width: end_width]

                            name = "icrop" + str(self.transformation_index)
                            self.transformation_index += 1
                            elemento = {(name, function): values}
                            new_test_data.append(elemento)
                self.test_data = new_test_data

                if self.dev_path:
                    n = len(self.dev_data)
                    new_dev_data = []
                    for index in range(n):
                        # Agrego los nuevos cortes de frames a los nuevos datos
                        values = tuple(self.dev_data[index].values())[0]
                        original_height = self.load_raw_frame(values[0][0]).shape[0]
                        original_width = self.load_raw_frame(values[0][0]).shape[1]
                        n_veces = min([original_width // self.frame_size[0], original_height // self.frame_size[1]])

                        for i in range(n_veces):
                            for j in range(n_veces):
                                start_width = i * self.frame_size[0]
                                end_width = start_width + self.frame_size[0]
                                start_height = j * self.frame_size[1]
                                end_height = start_height + self.frame_size[1]
                                function = lambda frame: frame[start_height: end_height, start_width: end_width]

                                name = "icrop" + str(self.transformation_index)
                                self.transformation_index += 1
                                elemento = {(name, function): values}
                                new_dev_data.append(elemento)
                    self.dev_data = new_dev_data

        elif mode == 'random':
            """Modo aleatorio, donde la funcion personalizada corresponde al numero 
                 de cortes aleatorio por frame que se vayan a hacer. Estos cortes son totalmente 
                 aleatorios desde el inicio hasta el limite donde se pueden recortar los 
                 frames."""
            if isinstance(custom_fn, int):
                n_veces = custom_fn
            else:
                raise ValueError(
                    'Al usar el modo de cortes de frames aleatorio, custom_fn debe ser un entero'
                    ', el valor entregado es de tipo: %s' % type(custom_fn)
                )
            if conserve_original:
                n = len(self.train_data)
                for index in range(n):
                    # Agrego los nuevos cortes de frames a los datos
                    values = tuple(self.train_data[index].values())[0]
                    original_height = self.load_raw_frame(values[0][0]).shape[0]
                    original_width = self.load_raw_frame(values[0][0]).shape[1]

                    for _ in range(n_veces):
                        start_width = np.random.randint(0, original_width - self.frame_size[0])
                        end_width = start_width + self.frame_size[0]
                        start_height =np.random.randint(0, original_height - self.frame_size[1])
                        end_height = start_height + self.frame_size[1]
                        function = lambda frame: frame[start_height: end_height, start_width: end_width]

                        name = "icrop" + str(self.transformation_index)
                        self.transformation_index += 1
                        elemento = {(name, function): values}
                        self.train_data.append(elemento)

                    # Reemplazo la funcion de los datos ya almacenados
                    llave_original = tuple(self.train_data[index].keys())[0]
                    llave_nueva = (llave_original[0] + "icrop" + str(self.transformation_index), self.resize_frame)
                    valores = tuple(self.train_data[index].values())[0]
                    self.train_data[index] = {llave_nueva: valores}

                n = len(self.test_data)
                for index in range(n):
                    # Agrego los nuevos cortes de frames a los datos
                    values = tuple(self.test_data[index].values())[0]
                    original_height = self.load_raw_frame(values[0][0]).shape[0]
                    original_width = self.load_raw_frame(values[0][0]).shape[1]

                    for _ in range(n_veces):
                        start_width = np.random.randint(0, original_width - self.frame_size[0])
                        end_width = start_width + self.frame_size[0]
                        start_height = np.random.randint(0, original_height - self.frame_size[1])
                        end_height = start_height + self.frame_size[1]
                        function = lambda frame: frame[start_height: end_height, start_width: end_width]

                        name = "icrop" + str(self.transformation_index)
                        self.transformation_index += 1
                        elemento = {(name, function): values}
                        self.test_data.append(elemento)

                    # Reemplazo la funcion de los datos ya almacenados
                    llave_original = tuple(self.test_data[index].keys())[0]
                    llave_nueva = (llave_original[0] + "icrop" + str(self.transformation_index), self.resize_frame)
                    valores = tuple(self.test_data[index].values())[0]
                    self.test_data[index] = {llave_nueva: valores}

                if self.dev_path:
                    n = len(self.dev_data)
                    for index in range(n):
                        # Agrego los nuevos cortes de frames a los datos
                        values = tuple(self.dev_data[index].values())[0]
                        original_height = self.load_raw_frame(values[0][0]).shape[0]
                        original_width = self.load_raw_frame(values[0][0]).shape[1]

                        for _ in range(n_veces):
                            start_width = np.random.randint(0, original_width - self.frame_size[0])
                            end_width = start_width + self.frame_size[0]
                            start_height = np.random.randint(0, original_height - self.frame_size[1])
                            end_height = start_height + self.frame_size[1]
                            function = lambda frame: frame[start_height: end_height, start_width: end_width]

                            name = "icrop" + str(self.transformation_index)
                            self.transformation_index += 1
                            elemento = {(name, function): values}
                            self.dev_data.append(elemento)

                        # Reemplazo la funcion de los datos ya almacenados
                        llave_original = tuple(self.dev_data[index].keys())[0]
                        llave_nueva = (llave_original[0] + "icrop" + str(self.transformation_index), self.resize_frame)
                        valores = tuple(self.dev_data[index].values())[0]
                        self.dev_data[index] = {llave_nueva: valores}
            else:
                n = len(self.train_data)
                new_train_data = []
                for index in range(n):
                    # Agrego los nuevos cortes de frames a los datos
                    values = tuple(self.train_data[index].values())[0]
                    original_height = self.load_raw_frame(values[0][0]).shape[0]
                    original_width = self.load_raw_frame(values[0][0]).shape[1]

                    for _ in range(n_veces):
                        start_width = np.random.randint(0, original_width - self.frame_size[0])
                        end_width = start_width + self.frame_size[0]
                        start_height = np.random.randint(0, original_height - self.frame_size[1])
                        end_height = start_height + self.frame_size[1]
                        function = lambda frame: frame[start_height: end_height, start_width: end_width]

                        name = "icrop" + str(self.transformation_index)
                        self.transformation_index += 1
                        elemento = {(name, function): values}
                        new_train_data.append(elemento)
                self.train_data = new_train_data

                n = len(self.test_data)
                new_test_data = []
                for index in range(n):
                    # Agrego los nuevos cortes de frames a los datos
                    values = tuple(self.test_data[index].values())[0]
                    original_height = self.load_raw_frame(values[0][0]).shape[0]
                    original_width = self.load_raw_frame(values[0][0]).shape[1]

                    for _ in range(n_veces):
                        start_width = np.random.randint(0, original_width - self.frame_size[0])
                        end_width = start_width + self.frame_size[0]
                        start_height = np.random.randint(0, original_height - self.frame_size[1])
                        end_height = start_height + self.frame_size[1]
                        function = lambda frame: frame[start_height: end_height, start_width: end_width]

                        name = "icrop" + str(self.transformation_index)
                        self.transformation_index += 1
                        elemento = {(name, function): values}
                        new_test_data.append(elemento)
                self.test_data = new_test_data

                if self.dev_path:
                    n = len(self.dev_data)
                    new_dev_data = []
                    for index in range(n):
                        # Agrego los nuevos cortes de frames a los datos
                        values = tuple(self.dev_data[index].values())[0]
                        original_height = self.load_raw_frame(values[0][0]).shape[0]
                        original_width = self.load_raw_frame(values[0][0]).shape[1]

                        for _ in range(n_veces):
                            start_width = np.random.randint(0, original_width - self.frame_size[0])
                            end_width = start_width + self.frame_size[0]
                            start_height = np.random.randint(0, original_height - self.frame_size[1])
                            end_height = start_height + self.frame_size[1]
                            function = lambda frame: frame[start_height: end_height, start_width: end_width]

                            name = "icrop" + str(self.transformation_index)
                            self.transformation_index += 1
                            elemento = {(name, function): values}
                            new_dev_data.append(elemento)
                    self.dev_data = new_dev_data

        elif mode == 'custom':
            """Metodo que se encarga de ejecutar la funcion customizada de corte 
            a cada frame de cada video."""
            if custom_fn:
                if conserve_original:
                    n = len(self.train_data)
                    for index in range(n):
                        #Arego lso nuevos cortes de franes a los datos
                        values = tuple(self.train_data[index].values())[0]
                        original_height = self.load_raw_frame(values[0][0]).shape[0]
                        original_width = self.load_raw_frame(values[0][0]).shape[1]
                        cortes = custom_fn(original_width, original_height)

                        try:
                            for corte in cortes:
                                size_frame = (corte[1] - corte[0], corte[3] - corte[2])
                                if size_frame[0] != self.frame_size[0] or size_frame[1] != self.frame_size[1]:
                                    raise ValueError(
                                        'El tamaño de los frames debe ser igual al tamaño '
                                        'especificado por el usuario. Tamaño encontrado de '
                                        '%s' % str(size_frame)
                                    )
                                function = lambda frame: frame[corte[2]: corte[3], corte[0]: corte[1]]
                                name = "icrop" + str(self.transformation_index)
                                self.transformation_index += 1
                                elemento = { (name, function) : values }
                                self.train_data.append(elemento)

                        except:
                            raise AttributeError(
                                'Se espera que la funcion customizada retorne una matriz '
                                'de forma que las filas es un corte a hacerle a cada video y '
                                'las columnas sean 4 (inicio corte x, fin corte x, inicio corte '
                                'y, fin corte y) exactamente en ese orden.'
                            )
                        # Reemplazo la funcion de los datos ya almacenados
                        llave_original = tuple(self.train_data[index].keys())[0]
                        llave_nueva = (llave_original[0] + "icrop" + str(self.transformation_index), self.resize_frame)
                        self.train_data[index] = {llave_nueva: values}

                    n = len(self.test_data)
                    for index in range(n):
                        # Arego lso nuevos cortes de franes a los datos
                        values = tuple(self.test_data[index].values())[0]
                        original_height = self.load_raw_frame(values[0][0]).shape[0]
                        original_width = self.load_raw_frame(values[0][0]).shape[1]
                        cortes = custom_fn(original_width, original_height)

                        try:
                            for corte in cortes:
                                size_frame = (corte[1] - corte[0], corte[3] - corte[2])
                                if size_frame[0] != self.frame_size[0] or size_frame[1] != self.frame_size[1]:
                                    raise ValueError(
                                        'El tamaño de los frames debe ser igual al tamaño '
                                        'especificado por el usuario. Tamaño encontrado de '
                                        '%s' % str(size_frame)
                                    )
                                function = lambda frame: frame[corte[2]: corte[3], corte[0]: corte[1]]
                                name = "icrop" + str(self.transformation_index)
                                self.transformation_index += 1
                                elemento = {(name, function): values}
                                self.test_data.append(elemento)

                        except:
                            raise AttributeError(
                                'Se espera que la funcion customizada retorne una matriz '
                                'de forma que las filas es un corte a hacerle a cada video y '
                                'las columnas sean 4 (inicio corte x, fin corte x, inicio corte '
                                'y, fin corte y) exactamente en ese orden.'
                            )
                        # Reemplazo la funcion de los datos ya almacenados
                        llave_original = tuple(self.test_data[index].keys())[0]
                        llave_nueva = (llave_original[0] + "icrop" + str(self.transformation_index), self.resize_frame)
                        self.test_data[index] = {llave_nueva: values}

                    if self.dev_path:
                        n = len(self.dev_data)
                        for index in range(n):
                            # Arego lso nuevos cortes de franes a los datos
                            values = tuple(self.dev_data[index].values())[0]
                            original_height = self.load_raw_frame(values[0][0]).shape[0]
                            original_width = self.load_raw_frame(values[0][0]).shape[1]
                            cortes = custom_fn(original_width, original_height)

                            try:
                                for corte in cortes:
                                    size_frame = (corte[1] - corte[0], corte[3] - corte[2])
                                    if size_frame[0] != self.frame_size[0] or size_frame[1] != self.frame_size[1]:
                                        raise ValueError(
                                            'El tamaño de los frames debe ser igual al tamaño '
                                            'especificado por el usuario. Tamaño encontrado de '
                                            '%s' % str(size_frame)
                                        )
                                    function = lambda frame: frame[corte[2]: corte[3], corte[0]: corte[1]]
                                    name = "icrop" + str(self.transformation_index)
                                    self.transformation_index += 1
                                    elemento = {(name, function): values}
                                    self.dev_data.append(elemento)

                            except:
                                raise AttributeError(
                                    'Se espera que la funcion customizada retorne una matriz '
                                    'de forma que las filas es un corte a hacerle a cada video y '
                                    'las columnas sean 4 (inicio corte x, fin corte x, inicio corte '
                                    'y, fin corte y) exactamente en ese orden.'
                                )
                            # Reemplazo la funcion de los datos ya almacenados
                            llave_original = tuple(self.dev_data[index].keys())[0]
                            llave_nueva = (
                            llave_original[0] + "icrop" + str(self.transformation_index), self.resize_frame)
                            self.dev_data[index] = {llave_nueva: values}

                else:
                    n = len(self.train_data)
                    new_train_data = []
                    for index in range(n):
                        # Reemplazo los nuevos cortes de frames a los datos
                        values = tuple(self.train_data[index].values())[0]
                        original_height = self.load_raw_frame(values[0][0]).shape[0]
                        original_width = self.load_raw_frame(values[0][0]).shape[1]
                        cortes = custom_fn(original_width, original_height)

                        try:
                            for corte in cortes:
                                size_frame = (corte[1] - corte[0], corte[3] - corte[2])
                                if size_frame[0] != self.frame_size[0] or size_frame[1] != self.frame_size[1]:
                                    raise ValueError(
                                        'El tamaño de los frames debe ser igual al tamaño '
                                        'especificado por el usuario. Tamaño encontrado de '
                                        '%s' % str(size_frame)
                                    )
                                function = lambda frame: frame[corte[2]: corte[3], corte[0]: corte[1]]
                                name = "icrop" + str(self.transformation_index)
                                self.transformation_index += 1
                                elemento = {(name, function): values}
                                new_train_data.append(elemento)

                        except:
                            raise AttributeError(
                                'Se espera que la funcion customizada retorne una matriz '
                                'de forma que las filas es un corte a hacerle a cada video y '
                                'las columnas sean 4 (inicio corte x, fin corte x, inicio corte '
                                'y, fin corte y) exactamente en ese orden.'
                            )
                    self.train_data = new_train_data

                    n = len(self.test_data)
                    new_test_data = []
                    for index in range(n):
                        # Reemplazo los nuevos cortes de frames a los datos
                        values = tuple(self.test_data[index].values())[0]
                        original_height = self.load_raw_frame(values[0][0]).shape[0]
                        original_width = self.load_raw_frame(values[0][0]).shape[1]
                        cortes = custom_fn(original_width, original_height)

                        try:
                            for corte in cortes:
                                size_frame = (corte[1] - corte[0], corte[3] - corte[2])
                                if size_frame[0] != self.frame_size[0] or size_frame[1] != self.frame_size[1]:
                                    raise ValueError(
                                        'El tamaño de los frames debe ser igual al tamaño '
                                        'especificado por el usuario. Tamaño encontrado de '
                                        '%s' % str(size_frame)
                                    )
                                function = lambda frame: frame[corte[2]: corte[3], corte[0]: corte[1]]
                                name = "icrop" + str(self.transformation_index)
                                self.transformation_index += 1
                                elemento = {(name, function): values}
                                new_test_data.append(elemento)

                        except:
                            raise AttributeError(
                                'Se espera que la funcion customizada retorne una matriz '
                                'de forma que las filas es un corte a hacerle a cada video y '
                                'las columnas sean 4 (inicio corte x, fin corte x, inicio corte '
                                'y, fin corte y) exactamente en ese orden.'
                            )
                    self.test_data = new_test_data

                    if self.dev_path:
                        n = len(self.dev_data)
                        new_dev_data = []
                        for index in range(n):
                            # Reemplazo los nuevos cortes de frames a los datos
                            values = tuple(self.dev_data[index].values())[0]
                            original_height = self.load_raw_frame(values[0][0]).shape[0]
                            original_width = self.load_raw_frame(values[0][0]).shape[1]
                            cortes = custom_fn(original_width, original_height)

                            try:
                                for corte in cortes:
                                    size_frame = (corte[1] - corte[0], corte[3] - corte[2])
                                    if size_frame[0] != self.frame_size[0] or size_frame[1] != self.frame_size[1]:
                                        raise ValueError(
                                            'El tamaño de los frames debe ser igual al tamaño '
                                            'especificado por el usuario. Tamaño encontrado de '
                                            '%s' % str(size_frame)
                                        )
                                    function = lambda frame: frame[corte[2]: corte[3], corte[0]: corte[1]]
                                    name = "icrop" + str(self.transformation_index)
                                    self.transformation_index += 1
                                    elemento = {(name, function): values}
                                    new_dev_data.append(elemento)

                            except:
                                raise AttributeError(
                                    'Se espera que la funcion customizada retorne una matriz '
                                    'de forma que las filas es un corte a hacerle a cada video y '
                                    'las columnas sean 4 (inicio corte x, fin corte x, inicio corte '
                                    'y, fin corte y) exactamente en ese orden.'
                                )
                        self.dev_data = new_dev_data

            else:
                raise ValueError('Debe pasar la funcion customizada para el '
                    'modo customizado, de lo contrario no podra usarlo. Tipo de dato'
                     ' de funcion customizada recibida: %s' % type(custom_fn))

        else:
            """Modo None, donde simplemente se redimensiona toda la imagen"""
            for index in range(len(self.train_data)):
                llave_original = tuple(self.train_data[index].keys())[0]
                llave_nueva = (llave_original[0] + "icrop" + str(self.transformation_index), self.resize_frame)
                valores = tuple(self.train_data[index].values())[0]
                self.train_data[index] = {llave_nueva: valores}

            for index in range(len(self.test_data)):
                llave_original = tuple(self.test_data[index].keys())[0]
                llave_nueva = (llave_original[0] + "icrop" + str(self.transformation_index), self.resize_frame)
                valores = tuple(self.test_data[index].values())[0]
                self.test_data[index] = {llave_nueva: valores}

            if self.dev_path:
                for index in range(len(self.dev_data)):
                    llave_original = tuple(self.dev_data[index].keys())[0]
                    llave_nueva = (llave_original[0] + "icrop" + str(self.transformation_index), self.resize_frame)
                    valores = tuple(self.dev_data[index].values())[0]
                    self.dev_data[index] = {llave_nueva: valores}
