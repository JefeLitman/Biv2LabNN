"""This file contain the blocks for models build in all _models.py
Version: 1.0
Made by: Edgar Rangel
"""

def get_n_blocks_conv(n_blocks:int, first_block_input:keras.layers.Layer, 
    conv_filters:list, conv_kernel_size:list, maxpool_kernel:list, 
    conv_padding:list, maxpool_padding:list, with_batch_norm:bool = False,
    layers_activation:str = "relu", with_bias:bool = True, 
    regularizers:tuple = (None, None), initializers:tuple = (None, None)):
    """Function that return the tensorflow graph (forward pipeline) with the parameters given. This 
    method apply 5 convolutional blocks to the current forward pipeline (first_block_input) and return 
    the output of last block. Finally the BatchNormalization layers are by default and no customizable.
    Args:
        first_block_input: An instance of tf.keras.layers.Layer with the input to the first block.
        with_batch_norm: A boolean indicating if apply BatchNormalization after the convolutional layer.
        layers_padding: String indicating how to do the padding in convolutional layers.
        layers_activation: String indicating what activation must be applied to convolutional layers.
        with_bias: A boolean indicating if the convolutional layers can have bias.
        regularizers: A tuple with two tf.keras.regularizers instances, the structure is (kernel_regularizer, 
            bias_regularizer). This apply over all the convolutional layers.
        initializers: A tuple with two tf.keras.initializers instances, the structure is (kernel_initializer, 
            bias_initializer). This apply over all the convolutional layers.
    """
    #Conv1
    x = keras.layers.Conv3D(filters=64, kernel_size=3, padding="same", activation="relu", 
                              kernel_regularizer=keras.regularizers.l2(weigh_decay),
                              name='conv3d_1')(first_block_input)
    x = keras.layers.BatchNormalization(name="batch_norm_1")(x)
    x = keras.layers.MaxPool3D(pool_size=(1,2,2),strides=(1,2,2), name='max_pooling3d_1')(x)

    #Conv2
    x = keras.layers.Conv3D(filters=128, kernel_size=3, padding="same", activation="relu", 
                          kernel_regularizer=keras.regularizers.l2(weigh_decay),
                          name='conv3d_2')(x)
    x = keras.layers.BatchNormalization(axis=4, name="batch_norm_2")(x)
    x = keras.layers.MaxPool3D(pool_size=(2,2,2),strides=(2,2,2), name='max_pooling3d_2')(x)

    #Conv3
    x = keras.layers.Conv3D(filters=256, kernel_size=3, padding="same", activation="relu", 
                          kernel_regularizer=keras.regularizers.l2(weigh_decay),
                          name='conv3d_3')(x)
    x = keras.layers.BatchNormalization(axis=4, name="batch_norm_3")(x)
    x = keras.layers.MaxPool3D(pool_size=(2,2,2),strides=(2,2,2),name='max_pooling3d_3')(x)

    #Conv4
    x = keras.layers.Conv3D(filters=256, kernel_size=3, padding="same", activation="relu", 
                          kernel_regularizer=keras.regularizers.l2(weigh_decay),
                          name='conv3d_4')(x)
    x = keras.layers.BatchNormalization(axis=4, name="batch_norm_4")(x)
    x = keras.layers.MaxPool3D(pool_size=(2,2,2),strides=(2,2,2),name='max_pooling3d_4')(x)

    #Conv5
    x = keras.layers.Conv3D(filters=256, kernel_size=3, padding="same", activation="relu", 
                          kernel_regularizer=keras.regularizers.l2(weigh_decay),
                          name='conv3d_5')(x)
    x = keras.layers.BatchNormalization(axis=4, name="batch_norm_5")(x)
    return keras.layers.MaxPool3D(pool_size=(2,2,2),strides=(2,2,2),name='max_pooling3d_5')(x)