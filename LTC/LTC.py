import tensorflow as tf

class LTC():
    """Modelo LTC implementado en TensorFlow"""

    def __init__(self, num_clases,
                 batch_size,
                 video_shape):
        """Constructor del modelo LTC
        Args:
            num_clases: El numero de total de clases que se van a clasificar.
            batch_size: Numero de elemento dentro de cada batch.
            video_shape: Arreglo de la forma [frames, heigh, width, channels] que corresponde a la forma de los videos.
        """
        self.x = tf.placeholder(dtype=tf.float32, shape= [batch_size]+video_shape, name="Entradas")
        self.y = tf.placeholder(dtype=tf.int64, shape= [batch_size], name="Etiquetas")
        self.num_clases = num_clases
        self.batch_size = batch_size

    def inicializar_modelo(self, learning_rate, dropout_rate = 0.5):
        """Metodo que se encarga de construir la grafica de tensorflow para el modelo.
        Args:
            learning_rate: Numero desde 0 que corresponde al paso que debe aplicar el optimizador
            dropout_rate: Decimal entre 0 y 1 donde corresponde a la tasa de dropout en la red en entrenamiento
            """
        self.entrenamiento = False #Variable que indica si esta en modo entrenamiento o no
        self.lr = learning_rate
        self.dropout_rate = dropout_rate
        self.__construir_modelo__()
        self.__construir_operaciones_gradientes__()

    def __construir_modelo__(self):
        """Metodo que se encarga de construir la grafica del modelo para poder """

        # Capa conv1
        conv1 = self.__convolucion_3d__('conv1', self.x, 3, self.x.shape[4], 64, self.__formateo_stride__(1, 1, 1))
        relu1 = self.__relu__('relu1', conv1)
        max_pool1 = self.__max_pooling_3d__('max_pool1', relu1, [1, 1, 2, 2, 1], self.__formateo_stride__(2, 2, 2))

        # Capa conv2
        conv2 = self.__convolucion_3d__('conv2', max_pool1, 3, 64, 128, self.__formateo_stride__(1, 1, 1))
        relu2 = self.__relu__('relu2', conv2)
        max_pool2 = self.__max_pooling_3d__('max_pool2', relu2, [1, 2, 2, 2, 1], self.__formateo_stride__(2, 2, 2))

        # Capa conv3
        conv3 = self.__convolucion_3d__('conv3', max_pool2, 3, 128, 256, self.__formateo_stride__(1, 1, 1))
        relu3 = self.__relu__('relu3', conv3)
        max_pool3 = self.__max_pooling_3d__('max_pool3', relu3, [1, 2, 2, 2, 1], self.__formateo_stride__(2, 2, 2))

        # Capa conv4
        conv4 = self.__convolucion_3d__('conv4', max_pool3, 3, 256, 256, self.__formateo_stride__(1, 1, 1))
        relu4 = self.__relu__('relu4', conv4)
        max_pool4 = self.__max_pooling_3d__('max_pool4', relu4, [1, 2, 2, 2, 1], self.__formateo_stride__(2, 2, 2))

        # Capa conv5
        conv5 = self.__convolucion_3d__('conv5', max_pool4, 3, 256, 256, self.__formateo_stride__(1, 1, 1))
        relu5 = self.__relu__('relu5', conv5)
        max_pool5 = self.__max_pooling_3d__('max_pool5', relu5, [1, 2, 2, 2, 1], self.__formateo_stride__(2, 2, 2))

        # Capa fc6
        flatten = self.__aplanar__("Aplanar6", max_pool5)
        fc6= self.__full_connected__('fc6', flatten, 2048)
        relu6 = self.__relu__('relu6', fc6)
        if self.entrenamiento:
            relu6 = tf.nn.dropout(relu6, rate=self.dropout_rate, name='dropout6')

        # Capa fc7
        fc7 = self.__full_connected__('fc7', relu6, 2048)
        relu7 = self.__relu__('relu7', fc7)
        if self.entrenamiento:
            relu7 = tf.nn.dropout(relu7, rate=self.dropout_rate, name='dropout7')

        # Capa fc8
        fc8 = self.__full_connected__('fc8', relu7, self.num_clases)
        self.predicciones = tf.nn.log_softmax(fc8, name='softmax8')
        predicciones = tf.argmax(self.predicciones, axis=1)

        # Computo del computo computacional
        with tf.variable_scope('costo'):
            negative_log_likehood = tf.losses.log_loss(labels=self.y,predictions=predicciones)
            self.perdida = tf.reduce_mean(negative_log_likehood)

            precision = tf.equal(self.y,predicciones)
            self.precision = tf.reduce_mean(tf.cast(precision, tf.float32))

            tf.summary.scalar('Loss', self.perdida)
            tf.summary.scalar('Accuracy',self.precision)

    def __construir_operaciones_gradientes__(self):
        """Metodo que se encarga de construir y aplicar los gradientes sobre las variables para
        entrenar."""

        lr = tf.constant(self.lr,dtype=tf.float32)

        optimizador = tf.train.GradientDescentOptimizer(lr)

        self.entrenar = optimizador.minimize(loss = self.predicciones, name="aplico_gradientes")

    def __formateo_stride__(self,depth,height,width):
        """Metodo que se encarga de retornar un arreglo para el stride segun el formato
        NDHWC.
        [batch, depth, Height, Width,  Channels]"""
        return [1,depth,height,width,1]

    def __convolucion_3d__(self, nombre_operacion, entrada, tam_kernel, filtros_entrada, filtros_salida, stride):
        """Metodo que me retorna la operacion de convolucion 3D en tensorflow
        Args:
            nombre_operacion: Correpsonde al nombre de la capa que tendra la convolucion 3D y debe ser unico
            tam_kernel: Numero que corresponde al tamaño del filtro que se va aplicar [depth, height, width, ins, outs]
            stride: Vector que contiene el stride que se va realizar sobre la convolucion [batch, depth, Height, Width,  Channels]
            """
        with tf.variable_scope(nombre_operacion):
            kernel = tf.get_variable(name="W",
                                     shape=[tam_kernel,tam_kernel,tam_kernel,filtros_entrada,filtros_salida],
                                     dtype=tf.float32,
                                     initializer=tf.glorot_uniform_initializer)
            return tf.nn.conv3d(input=entrada,
                                filter=kernel,
                                strides=stride,
                                padding='SAME',
                                name=nombre_operacion)

    def __max_pooling_3d__(self,nombre_operacion, entrada,tam_kernel,stride):
        """Metodo que me retorna la operacion de max_pooling en 3D
        Args:
            nombre_operacion: Correpsonde al nombre de la capa que tendra la convolucion 3D y debe ser unico
            tam_kernel: Vector que corresponde al tamaño de kernel que se aplicara [batches, depth, height, width, channels]
            stride: Vector que contiene el stride que se va realizar sobre la convolucion [batch, depth, Height, Width,  Channels]
            """
        return tf.nn.max_pool3d(input=entrada,
                                    ksize=tam_kernel,
                                    strides=stride,
                                    padding='SAME',
                                    name=nombre_operacion)

    def __relu__(self,nombre_operacion,entrada):
        return  tf.nn.relu(features=entrada,
                           name=nombre_operacion)

    def __full_connected__(self,nombre_operacion,entrada,num_salidas):
        """Metodo que me retorna una capa full connected
        Args:
            nombre_operacion: Corresponde al nombre de la capa que tendra la full connected y debe ser unico
            entrada: Debe tener dimension [batch, num_entradas]
            num_entradas: Numero de entradas que tendra la red
            num_salidas: Numero de neuronas de la capa full connected
            """
        with tf.variable_scope(nombre_operacion):
            w = tf.get_variable("W",
                                shape=[entrada.shape[1].value,num_salidas],
                                dtype=tf.float32,
                                initializer=tf.glorot_uniform_initializer)
            b = tf.get_variable("B",
                                shape=num_salidas,
                                dtype=tf.float32,
                                initializer=tf.zeros_initializer)
            return tf.nn.xw_plus_b(entrada,
                                   weights=w,
                                   biases=b,
                                   name = nombre_operacion)

    def __aplanar__(self, nombre_operacion,entrada):
        return tf.reshape(tensor=entrada,
                          shape=[self.batch_size,-1],
                          name=nombre_operacion)

    def enable_training(self):
        self.entrenamiento = True

    def disable_training(self):
        self.entrenamiento = False