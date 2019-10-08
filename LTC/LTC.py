import tensorflow as tf

class LTC():
    """Modelo LTC implementado en TensorFlow"""

    def __init__(self,
                 entrada,
                 etiquetas,
                 num_clases,
                 batch_size = None,
                 dropout = None,
                 entramiento = False):
        """Constructor del modelo LTC
        Args:
            entrada: Los valores que le voy a  pasar a la red. [ batch, time, height, width,channels]
            etiquetas: Las etiquetas correspondientes a los valores de entrada. [batch, classes]
            entramiento: Define si el modelo estara en modo entrenamiento o ejecucion.
            num_clases: El numero de total de clases del dataset especificado
        """
        self.x = entrada
        self.y = etiquetas
        self.batch_size = batch_size
        self.num_clases = num_clases
        self.entrenamiento = entramiento
        self.dropout_rate = dropout

    def init_grafica(self,learning_rate):
        """Metodo que se encarga de construir la grafica de tensorflow para el modelo
        Tambien sirve para hacer las iteraciones (probablemente)"""
        self.global_step = tf.train.get_or_create_global_step() #Me sirve para el caso de entrenamiento contabilizar las iteraciones
        self.construir_modelo()
        if self.entrenamiento:
            self.construir_operaciones_gradientes(learning_rate)

    def construir_modelo(self):
        """Metodo que se encarga de construir el nucleo del modelo"""

        #Capa conv1
        self.x = self.convolucion_3d('conv1',self.x,3,5,64,self.formateo_stride(1,1,1))
        self.x = self.relu('relu1',self.x)
        self.x = self.max_pooling_3d('max_pool1',self.x,[1,1,2,2,1],self.formateo_stride(2,2,2))

        # Capa conv2
        self.x = self.convolucion_3d('conv2', self.x, 3, 64, 128, self.formateo_stride(1, 1, 1))
        self.x = self.relu('relu2', self.x)
        self.x = self.max_pooling_3d('max_pool2', self.x, [1, 2, 2, 2, 1], self.formateo_stride(2, 2, 2))

        # Capa conv3
        self.x = self.convolucion_3d('conv3', self.x, 3, 128, 256, self.formateo_stride(1, 1, 1))
        self.x = self.relu('relu3', self.x)
        self.x = self.max_pooling_3d('max_pool3', self.x, [1, 2, 2, 2, 1], self.formateo_stride(2, 2, 2))

        # Capa conv4
        self.x = self.convolucion_3d('conv4', self.x, 3, 256, 256, self.formateo_stride(1, 1, 1))
        self.x = self.relu('relu4', self.x)
        self.x = self.max_pooling_3d('max_pool4', self.x, [1, 2, 2, 2, 1], self.formateo_stride(2, 2, 2))

        # Capa conv5
        self.x = self.convolucion_3d('conv5', self.x, 3, 256, 256, self.formateo_stride(1, 1, 1))
        self.x = self.relu('relu5', self.x)
        self.x = self.max_pooling_3d('max_pool5', self.x, [1, 2, 2, 2, 1], self.formateo_stride(2, 2, 2))

        #Capa fc6
        self.x = self.aplanar("Aplanar6",self.x)
        self.x = self.full_connected('fc6',self.x,2048)
        self.x = self.relu('relu6',self.x)
        if self.entrenamiento:
            self.x = tf.nn.dropout(self.x,rate=self.dropout_rate,name='dropout6')

        # Capa fc7
        self.x = self.full_connected('fc7', self.x, 2048)
        self.x = self.relu('relu7', self.x)
        if self.entrenamiento:
            self.x = tf.nn.dropout(self.x, rate=self.dropout_rate, name='dropout7')

        # Capa fc8
        self.x = self.full_connected('fc8', self.x, self.num_clases)
        self.predicciones = tf.nn.log_softmax(self.x,name='softmax8')

        #Computo del computo computacional
        with tf.variable_scope('costo'):
            negative_log_likehood = tf.losses.log_loss(self.y,self.predicciones)
            self.perdida = tf.reduce_mean(negative_log_likehood)
            tf.summary.scalar('Loss', self.perdida)

    def construir_operaciones_gradientes(self,learning_rate):
        """Metodo que se encarga de aplicar los gradientes y el aprendizaje en la red si el parametro
        de entrenamiento esta activo"""
        self.lr = tf.constant(learning_rate,dtype=tf.float32)

        optimizador = tf.train.GradientDescentOptimizer(self.lr)

        self.train_op = optimizador.minimize(self.perdida,global_step=self.global_step,name="paso_aprendizaje")

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
                          shape=[self.batch_size,-1],
                          name=nombre_operacion)