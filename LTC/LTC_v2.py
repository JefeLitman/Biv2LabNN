import tensorflow as tf

class LTC():
    """Modelo LTC implementado en TensorFlow"""

    def __init__(self, num_clases):
        """Constructor del modelo LTC
        Args:
            num_clases: El numero de total de clases del dataset especificado
        """
        self.num_clases = num_clases
        self.perdidas = []
        self.precisiones = []

    def propagacion(self, entrada,
                    etiquetas,
                    dropout_rate = 0.5,
                    entrenar = False):
        """Metodo que se encarga de realizar la propagacion de una entrada determinada
        sobre toda la arquitectura de la red.
        Args:
            entrada: Tensor de dimensiones [N,D,H,W,C]
            etiquetas: Tensor de dimensones [N,]
            dropout_rate: Decimal entre 0 y 1 donde corresponde a la tasa de dropout en la red en entrenamiento
            entrenar: Booleano que indica si la red esta en modo de aprendizaje o de validacion.
            """

        # Capa conv1
        x = self.convolucion_3d('conv1', entrada, 3, entrada.shape[4], 64, self.formateo_stride(1, 1, 1))
        x = self.relu('relu1', x)
        x = self.max_pooling_3d('max_pool1', x, [1, 1, 2, 2, 1], self.formateo_stride(2, 2, 2))

        # Capa conv2
        x = self.convolucion_3d('conv2', x, 3, 64, 128, self.formateo_stride(1, 1, 1))
        x = self.relu('relu2', x)
        x = self.max_pooling_3d('max_pool2', x, [1, 2, 2, 2, 1], self.formateo_stride(2, 2, 2))

        # Capa conv3
        x = self.convolucion_3d('conv3', x, 3, 128, 256, self.formateo_stride(1, 1, 1))
        x = self.relu('relu3', x)
        x = self.max_pooling_3d('max_pool3', x, [1, 2, 2, 2, 1], self.formateo_stride(2, 2, 2))

        # Capa conv4
        x = self.convolucion_3d('conv4', x, 3, 256, 256, self.formateo_stride(1, 1, 1))
        x = self.relu('relu4', x)
        x = self.max_pooling_3d('max_pool4', x, [1, 2, 2, 2, 1], self.formateo_stride(2, 2, 2))

        # Capa conv5
        x = self.convolucion_3d('conv5', x, 3, 256, 256, self.formateo_stride(1, 1, 1))
        x = self.relu('relu5', x)
        x = self.max_pooling_3d('max_pool5', x, [1, 2, 2, 2, 1], self.formateo_stride(2, 2, 2))

        # Capa fc6
        x = self.aplanar("Aplanar6", x)
        x = self.full_connected('fc6', x, 2048)
        x = self.relu('relu6', x)
        if entrenar:
            x = tf.nn.dropout(x, rate=dropout_rate, name='dropout6')

        # Capa fc7
        x = self.full_connected('fc7', x, 2048)
        x = self.relu('relu7', x)
        if entrenar:
            x = tf.nn.dropout(x, rate=dropout_rate, name='dropout7')

        # Capa fc8
        x = self.full_connected('fc8', x, self.num_clases)
        predicciones = tf.nn.log_softmax(x, name='softmax8')
        predicciones = tf.argmax(predicciones, axis=1)

        # Computo del computo computacional
        with tf.variable_scope('costo'):
            negative_log_likehood = tf.losses.log_loss(etiquetas, predicciones)
            perdida = tf.reduce_mean(negative_log_likehood)
            self.perdidas.append(perdida)

            precision = tf.equal(etiquetas,predicciones)
            precision = tf.reduce_mean(tf.cast(precision, tf.float32))
            self.precisiones.append(precision)

            tf.summary.scalar('Loss', perdida)
            tf.summary.scalar('Accuracy',precision)

        return predicciones, precision, perdida

    def entrenar(self,entrada, etiquetas, dropout, learning_rate):
        """Metodo que se encarga de realizar el aprendizaje, es decir propagar, computar y aplicar
        los gradientes en el modelo. Unicamente se aplica si se esta entrenando.
        Args:
            entrada: Tensor de dimensiones [N,D,H,W,C]
            etiquetas: Tensor de dimensones [N,]
            dropout: Decimal entre 0 y 1 donde corresponde a la tasa de dropout para la red
            learning_rate: Decimal que corresponde al paso que debe dar el optimizador a aplicar
            """

        _, _, perdida = self.propagacion(entrada=entrada, etiquetas=etiquetas, dropout_rate=dropout, entrenar=True)

        lr = tf.constant(learning_rate,dtype=tf.float32)

        optimizador = tf.train.GradientDescentOptimizer(lr)

        return optimizador.minimize(perdida,name="un_paso_aprendizaje")

    def formateo_stride(self,depth,height,width):
        """Metodo que se encarga de retornar un arreglo para el stride segun el formato
        NDHWC.
        [batch, depth, Height, Width,  Channels]"""
        return [1,depth,height,width,1]

    def convolucion_3d(self, nombre_operacion, entrada, tam_kernel, filtros_entrada, filtros_salida, stride):
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
                                stride=stride,
                                padding='SAME',
                                name=nombre_operacion)

    def max_pooling_3d(self,nombre_operacion, entrada,tam_kernel,stride):
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

    def relu(self,nombre_operacion,entrada):
        return  tf.nn.relu(features=entrada,
                           name=nombre_operacion)

    def full_connected(self,nombre_operacion,entrada,num_salidas):
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

    def aplanar(self, nombre_operacion,entrada):
        return tf.reshape(tensor=entrada,
                          shape=[entrada.shape[0],-1],
                          name=nombre_operacion)