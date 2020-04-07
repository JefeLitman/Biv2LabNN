from tensorflow import keras
from inception_modules import *
from encoders import *

def get_convs_blocks5(inputs, weigh_decay):
    #Conv1
    x = keras.layers.Conv3D(filters=64, kernel_size=3, padding="same", activation="relu", 
                              kernel_regularizer=keras.regularizers.l2(weigh_decay),
                              name='conv3d_1')(inputs)
    x = keras.layers.BatchNormalization(axis=4, name="batch_norm_1")(x)
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

def get_convs_blocks4(inputs, weigh_decay):
    #Conv1
    x = keras.layers.Conv3D(filters=64, kernel_size=3, padding="same", activation="relu", 
                              kernel_regularizer=keras.regularizers.l2(weigh_decay),
                              name='conv3d_1')(inputs)
    x = keras.layers.BatchNormalization(axis=4, name="batch_norm_1")(x)
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
    return keras.layers.MaxPool3D(pool_size=(2,2,2),strides=(2,2,2),name='max_pooling3d_4')(x)

def get_convs_blocks5_noMaxPoolTemporal(inputs, weigh_decay):
    #Conv1
    x = keras.layers.Conv3D(filters=64, kernel_size=3, padding="same", activation="relu", 
                              kernel_regularizer=keras.regularizers.l2(weigh_decay),
                              name='conv3d_1')(inputs)
    x = keras.layers.BatchNormalization(axis=4, name="batch_norm_1")(x)
    x = keras.layers.MaxPool3D(pool_size=(1,2,2),strides=(1,2,2), name='max_pooling3d_1')(x)

    #Conv2
    x = keras.layers.Conv3D(filters=128, kernel_size=3, padding="same", activation="relu", 
                          kernel_regularizer=keras.regularizers.l2(weigh_decay),
                          name='conv3d_2')(x)
    x = keras.layers.BatchNormalization(axis=4, name="batch_norm_2")(x)
    x = keras.layers.MaxPool3D(pool_size=(2,2,2),strides=(1,2,2), name='max_pooling3d_2')(x)

    #Conv3
    x = keras.layers.Conv3D(filters=256, kernel_size=3, padding="same", activation="relu", 
                          kernel_regularizer=keras.regularizers.l2(weigh_decay),
                          name='conv3d_3')(x)
    x = keras.layers.BatchNormalization(axis=4, name="batch_norm_3")(x)
    x = keras.layers.MaxPool3D(pool_size=(2,2,2),strides=(1,2,2),name='max_pooling3d_3')(x)

    #Conv4
    x = keras.layers.Conv3D(filters=256, kernel_size=3, padding="same", activation="relu", 
                          kernel_regularizer=keras.regularizers.l2(weigh_decay),
                          name='conv3d_4')(x)
    x = keras.layers.BatchNormalization(axis=4, name="batch_norm_4")(x)
    x = keras.layers.MaxPool3D(pool_size=(2,2,2),strides=(1,2,2),name='max_pooling3d_4')(x)

    #Conv5
    x = keras.layers.Conv3D(filters=256, kernel_size=3, padding="same", activation="relu", 
                          kernel_regularizer=keras.regularizers.l2(weigh_decay),
                          name='conv3d_5')(x)
    x = keras.layers.BatchNormalization(axis=4, name="batch_norm_5")(x)
    return keras.layers.MaxPool3D(pool_size=(2,2,2),strides=(1,2,2),name='max_pooling3d_5')(x)

def get_full_connecteds(inputs, n_classes, dropout, weigh_decay, initial_layer):
    #fc1s
    x = keras.layers.Flatten(name='flatten_'+str(initial_layer))(inputs)
    x = keras.layers.Dense(2048, activation="relu", kernel_regularizer=keras.regularizers.l2(weigh_decay),
                         name='dense_'+str(initial_layer))(x)
    x = keras.layers.Dropout(rate=dropout,name='dropout_'+str(initial_layer))(x)

    #fc2
    x = keras.layers.Dense(2048, activation="relu", kernel_regularizer=keras.regularizers.l2(weigh_decay),
                         name='dense_'+str(initial_layer+1))(x)
    x = keras.layers.Dropout(rate=dropout,name='dropout_'+str(initial_layer+1))(x)

    #fc3
    return keras.layers.Dense(n_classes, activation="softmax", 
                              kernel_regularizer=keras.regularizers.l2(weigh_decay), name='dense_'+str(initial_layer+2))(x)

def get_LTC_encoder_basic_I_noMaxPool(video_shape, n_classes, dropout, weigh_decay, cell_units, channels_reduction, 
                            with_sequences):
    entrada = keras.Input(shape=video_shape,
                     name="Input_video")

    x = get_convs_blocks5_noMaxPoolTemporal(entrada, weigh_decay)

    #Inception module
    x = inception_enhanced(x, 6, channels_reduction)

    #Encoder module
    x = basic_encoder(x, cell_units, 7, with_sequences)

    salidas = get_full_connecteds(x, n_classes,dropout,weigh_decay, 8)

    return keras.Model(entrada, salidas, name="LTC_BactchNorm_BasicEnc_Inc_noMaxPool")

def get_LTC_encoder_bidirec_I_noMaxPool(video_shape, n_classes, dropout, weigh_decay, cell_units, channels_reduction, 
                            with_sequences):
    entrada = keras.Input(shape=video_shape,
                     name="Input_video")

    x = get_convs_blocks5_noMaxPoolTemporal(entrada, weigh_decay)

    #Inception module
    x = inception_enhanced(x, 6, channels_reduction)

    #Encoder module
    x = unique_bidirectional_encoder(x, cell_units, 7, with_sequences)

    salidas = get_full_connecteds(x, n_classes,dropout,weigh_decay, 8)

    return keras.Model(entrada, salidas, name="LTC_BactchNorm_BiDirEnc_Inc_noMaxPool")

def get_LTC_encoder_slt_I_noMaxPool(video_shape, n_classes, dropout, weigh_decay, cell_units, channels_reduction, 
                        with_sequences):
    entrada = keras.Input(shape=video_shape,
                     name="Input_video")

    x = get_convs_blocks5_noMaxPoolTemporal(entrada, weigh_decay)

    #Inception module
    x = inception_enhanced(x, 6, channels_reduction)

    #Encoder module
    x = jota_encoder_slt(x, cell_units, 7, with_sequences)

    salidas = get_full_connecteds(x, n_classes,dropout,weigh_decay, 8)

    return keras.Model(entrada, salidas, name="LTC_BactchNorm_SLTEnc_Inc_noMaxPool")

def get_LTC_encoder_basic_NI_noMaxPool(video_shape, n_classes, dropout, weigh_decay, cell_units, with_sequences):
    entrada = keras.Input(shape=video_shape,
                     name="Input_video")

    x = get_convs_blocks5_noMaxPoolTemporal(entrada, weigh_decay)

    #Encoder module
    x = basic_encoder(x, cell_units, 6, with_sequences)

    salidas = get_full_connecteds(x, n_classes,dropout,weigh_decay, 7)

    return keras.Model(entrada, salidas, name="LTC_BactchNorm_BasicEnc_NoInc_noMaxPool")

def get_LTC_encoder_bidirec_NI_noMaxPool(video_shape, n_classes, dropout, weigh_decay, cell_units, with_sequences):
    entrada = keras.Input(shape=video_shape,
                     name="Input_video")

    x = get_convs_blocks5_noMaxPoolTemporal(entrada, weigh_decay)

    #Encoder module
    x = unique_bidirectional_encoder(x, cell_units, 6, with_sequences)

    salidas = get_full_connecteds(x, n_classes,dropout,weigh_decay, 7)

    return keras.Model(entrada, salidas, name="LTC_BactchNorm_BiDirEnc_NoInc_noMaxPool")

def get_LTC_encoder_slt_NI_noMaxPool(video_shape, n_classes, dropout, weigh_decay, cell_units, with_sequences):
    entrada = keras.Input(shape=video_shape,
                     name="Input_video")

    x = get_convs_blocks5_noMaxPoolTemporal(entrada, weigh_decay)

    #Encoder module
    x = jota_encoder_slt(x, cell_units, 6, with_sequences)

    salidas = get_full_connecteds(x, n_classes,dropout,weigh_decay, 7)

    return keras.Model(entrada, salidas, name="LTC_BactchNorm_SLTEnc_NoInc_noMaxPool")

def get_LTC_encoder_basic_I(video_shape, n_classes, dropout, weigh_decay, cell_units, channels_reduction, 
                            with_sequences):
    entrada = keras.Input(shape=video_shape,
                     name="Input_video")

    x = get_convs_blocks5(entrada, weigh_decay)

    #Inception module
    x = inception_enhanced(x, 6, channels_reduction)

    #Encoder module
    x = basic_encoder(x, cell_units, 7, with_sequences)

    salidas = get_full_connecteds(x, n_classes,dropout,weigh_decay, 8)

    return keras.Model(entrada, salidas, name="LTC_BactchNorm_BasicEnc_Inc")

def get_LTC_encoder_bidirec_I(video_shape, n_classes, dropout, weigh_decay, cell_units, channels_reduction, 
                            with_sequences):
    entrada = keras.Input(shape=video_shape,
                     name="Input_video")

    x = get_convs_blocks5(entrada, weigh_decay)

    #Inception module
    x = inception_enhanced(x, 6, channels_reduction)

    #Encoder module
    x = unique_bidirectional_encoder(x, cell_units, 7, with_sequences)

    salidas = get_full_connecteds(x, n_classes,dropout,weigh_decay, 8)

    return keras.Model(entrada, salidas, name="LTC_BactchNorm_BiDirEnc_Inc")

def get_LTC_encoder_slt_I(video_shape, n_classes, dropout, weigh_decay, cell_units, channels_reduction, 
                        with_sequences):
    entrada = keras.Input(shape=video_shape,
                     name="Input_video")

    x = get_convs_blocks5(entrada, weigh_decay)

    #Inception module
    x = inception_enhanced(x, 6, channels_reduction)

    #Encoder module
    x = jota_encoder_slt(x, cell_units, 7, with_sequences)

    salidas = get_full_connecteds(x, n_classes,dropout,weigh_decay, 8)

    return keras.Model(entrada, salidas, name="LTC_BactchNorm_SLTEnc_Inc")

def get_LTC_encoder_basic_NI(video_shape, n_classes, dropout, weigh_decay, cell_units, with_sequences):
    entrada = keras.Input(shape=video_shape,
                     name="Input_video")

    x = get_convs_blocks5(entrada, weigh_decay)

    #Encoder module
    x = basic_encoder(x, cell_units, 6, with_sequences)

    salidas = get_full_connecteds(x, n_classes,dropout,weigh_decay, 7)

    return keras.Model(entrada, salidas, name="LTC_BactchNorm_BasicEnc_NoInc")

def get_LTC_encoder_bidirec_NI(video_shape, n_classes, dropout, weigh_decay, cell_units, with_sequences):
    entrada = keras.Input(shape=video_shape,
                     name="Input_video")

    x = get_convs_blocks5(entrada, weigh_decay)

    #Encoder module
    x = unique_bidirectional_encoder(x, cell_units, 6, with_sequences)

    salidas = get_full_connecteds(x, n_classes,dropout,weigh_decay, 7)

    return keras.Model(entrada, salidas, name="LTC_BactchNorm_BiDirEnc_NoInc")

def get_LTC_encoder_slt_NI(video_shape, n_classes, dropout, weigh_decay, cell_units, with_sequences):
    entrada = keras.Input(shape=video_shape,
                     name="Input_video")

    x = get_convs_blocks5(entrada, weigh_decay)

    #Encoder module
    x = jota_encoder_slt(x, cell_units, 6, with_sequences)

    salidas = get_full_connecteds(x, n_classes,dropout,weigh_decay, 7)

    return keras.Model(entrada, salidas, name="LTC_BactchNorm_SLTEnc_NoInc")

def get_LTC_inception_enhan_minus1(video_shape, n_classes, dropout, weigh_decay, channels_reduction):
    entrada = keras.Input(shape=video_shape,
                     name="Input_video")
    x = get_convs_blocks4(entrada, weigh_decay)

    #Inception Module
    x = inception_enhanced(x, 5, channels_reduction)

    salidas = get_full_connecteds(x, n_classes, dropout, weigh_decay, 6)

    return keras.Model(entrada, salidas, name="LTC_BatchNorm_inception-enhanced-minus1")

def get_LTC_inception_naive_minus1(video_shape, n_classes, dropout, weigh_decay, channels_reduction):
    entrada = keras.Input(shape=video_shape,
                     name="Input_video")
    x = get_convs_blocks4(entrada, weigh_decay)

    #Inception Module
    x = inception_naive(x, 5, channels_reduction)

    salidas = get_full_connecteds(x, n_classes, dropout, weigh_decay, 6)

    return keras.Model(entrada, salidas, name="LTC_BatchNorm_inception-naive-minus1")

def get_LTC_inception_enhanced(video_shape, n_classes, dropout, weigh_decay, channels_reduction):
    entrada = keras.Input(shape=video_shape,
                     name="Input_video")
    
    x = get_convs_blocks5(entrada, weigh_decay)

    #Inception Module
    x = inception_enhanced(x, 6, channels_reduction)

    salidas = get_full_connecteds(x, n_classes, dropout, weigh_decay, 7)

    return keras.Model(entrada, salidas, name="LTC_BatchNorm_inception-enhanced")

def get_LTC_inception_naive(video_shape, n_classes, dropout, weigh_decay, channels_reduction):
    entrada = keras.Input(shape=video_shape,
                     name="Input_video")
    x = get_convs_blocks5(entrada)

    #Inception Module
    x = inception_naive(x, 6, channels_reduction)

    salidas = get_full_connecteds(x, n_classes, dropout, weigh_decay, 7)

    return keras.Model(entrada, salidas, name="LTC_BatchNorm_inception-naive")

def get_LTC_inception_convTrans3D(video_shape, n_classes, dropout, weigh_decay, channels_reduction):
    entrada = keras.Input(shape=video_shape,
                     name="Input_video")
    x = get_convs_blocks5(x, weigh_decay)

    #Inception Module
    x = convTrans3D(x, 6, channels_reduction)

    salidas = get_full_connecteds(x, n_classes, dropout, weigh_decay, 7)

    return keras.Model(entrada, salidas, name="LTC_BatchNorm_convTrans3D")

def get_LTC_inception_channels(video_shape, n_classes, dropout, weigh_decay, channels_reduction):
    entrada = keras.Input(shape=video_shape,
                     name="Input_video")
    x = get_convs_blocks5(entrada, weigh_decay)

    #Inception Module
    x = conv3D_Channels(x, 6, channels_reduction)

    salidas = get_full_connecteds(x, n_classes, dropout, weigh_decay, 7)

    return keras.Model(entrada, salidas, name="LTC_BatchNorm_inception-channels")

def get_LTC_BatchNorm(video_shape, n_classes, dropout, weigh_decay):
    entrada = keras.Input(shape=video_shape,
                         name="Input_video")

    x = get_convs_blocks5(entrada, weigh_decay)

    salidas = get_full_connecteds(x, n_classes, dropout, weigh_decay, 6)

    return keras.Model(entrada, salidas, name="LTC_BatchNorm")

def get_LTC_original(video_shape, n_classes, dropout, weigh_decay):
    """Function that return the original LTC model with the options given.
    Args:
        video_shape: Tuple or List with the shape of the input.
        n_classes: Integer with the number of classes to predict.
        dropout: Float between 0 and 1 for the dropout in the fully connected layers.
        weigh_decay: Float between 0 and inf for the weight decay in the layers.
    """

    entrada = keras.Input(shape=video_shape,
                         name="Input_video")
    #Conv1
    x = keras.layers.Conv3D(filters=64, kernel_size=3, padding="same", activation="relu", 
                              kernel_regularizer=keras.regularizers.l2(weigh_decay),
                              name='conv3d_1')(entrada)
    x = keras.layers.MaxPool3D(pool_size=(1,2,2),strides=(1,2,2), name='max_pooling3d_1')(x)

    #Conv2
    x = keras.layers.Conv3D(filters=128, kernel_size=3, padding="same", activation="relu", 
                          kernel_regularizer=keras.regularizers.l2(weigh_decay),
                          name='conv3d_2')(x)
    x = keras.layers.MaxPool3D(pool_size=(2,2,2),strides=(2,2,2), name='max_pooling3d_2')(x)

    #Conv3
    x = keras.layers.Conv3D(filters=256, kernel_size=3, padding="same", activation="relu", 
                          kernel_regularizer=keras.regularizers.l2(weigh_decay),
                          name='conv3d_3')(x)
    x = keras.layers.MaxPool3D(pool_size=(2,2,2),strides=(2,2,2),name='max_pooling3d_3')(x)

    #Conv4
    x = keras.layers.Conv3D(filters=256, kernel_size=3, padding="same", activation="relu", 
                          kernel_regularizer=keras.regularizers.l2(weigh_decay),
                          name='conv3d_4')(x)
    x = keras.layers.MaxPool3D(pool_size=(2,2,2),strides=(2,2,2),name='max_pooling3d_4')(x)

    #Conv5
    x = keras.layers.Conv3D(filters=256, kernel_size=3, padding="same", activation="relu", 
                          kernel_regularizer=keras.regularizers.l2(weigh_decay),
                          name='conv3d_5')(x)
    x = keras.layers.MaxPool3D(pool_size=(2,2,2),strides=(2,2,2),name='max_pooling3d_5')(x)

    #fc6s
    x = keras.layers.Flatten(name='flatten_6')(x)
    x = keras.layers.Dense(2048, activation="relu", kernel_regularizer=keras.regularizers.l2(weigh_decay),
                         name='dense_6')(x)
    x = keras.layers.Dropout(rate=dropout,name='dropout_6')(x)

    #fc7
    x = keras.layers.Dense(2048, activation="relu", kernel_regularizer=keras.regularizers.l2(weigh_decay),
                         name='dense_7')(x)
    x = keras.layers.Dropout(rate=dropout,name='dropout_7')(x)

    #fc8
    salidas = keras.layers.Dense(n_classes, activation="softmax", 
                              kernel_regularizer=keras.regularizers.l2(weigh_decay), name='dense_8')(x)

    return keras.Model(entrada, salidas, name="LTC_original")