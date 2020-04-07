from tensorflow import keras

def inception_naive(inputs, n_layer, output_filters):
    if output_filters < inputs.shape[-1]:
        raise ValueError("Is not possible to reduce the filters with the inception a module, because it have a maxpool layer")
    
    remaining_filters = output_filters - inputs.shape[-1]
    if remaining_filters % 2 != 0:
        remaining_filters += 1
    x1 = keras.layers.Conv3D(filters=remaining_filters//2, kernel_size=1,padding="same",
                             activation="relu",name='incep-1_V1-1x1x1_'+str(n_layer))(inputs)
    x2 = keras.layers.Conv3D(filters=remaining_filters//4, kernel_size=3,padding="same",
                             activation="relu",name='incep-2_V1-3x3x3_'+str(n_layer))(inputs)
    x3 = keras.layers.Conv3D(filters=remaining_filters//4, kernel_size=5,padding="same",
                             activation="relu",name='incep-3_V1-5x5x5_'+str(n_layer))(inputs)
    x4 = keras.layers.MaxPooling3D(pool_size=(1,1,1), strides=(1,1,1), 
                                   name='max_pool3d_incep-4_V1_1x1x1_'+str(n_layer))(inputs)
    
    return keras.layers.Concatenate(axis=-1, name="concat_incep_V1_"+str(n_layer))([x1,x2,x3,x4])

def inception_enhanced(inputs, n_layer, output_filters):
    if output_filters < inputs.shape[-1]:
        raise ValueError("Is not possible to reduce the filters with the inception b module, because it have a maxpool layer")
        
    remaining_filters = output_filters - inputs.shape[-1]
    if remaining_filters % 2 != 0:
        remaining_filters += 1
    #First layer
    x1 = keras.layers.Conv3D(filters=remaining_filters//2, kernel_size=1,padding="same",
                             activation="relu",name='incep-1_V2-1x1x1_'+str(n_layer)+'_0')(inputs)
    x2 = keras.layers.Conv3D(filters=remaining_filters//4, kernel_size=1,padding="same",
                             activation="relu",name='incep-2_V2-1x1x1_'+str(n_layer)+'_0')(inputs)
    x3 = keras.layers.Conv3D(filters=remaining_filters//4, kernel_size=1,padding="same",
                             activation="relu",name='incep-3_V2-1x1x1_'+str(n_layer)+'_0')(inputs)
    x4 = keras.layers.MaxPooling3D(pool_size=(1,1,1), strides=(1,1,1), 
                                   name='max_pool3d_incep-4_V2_1x1x1_'+str(n_layer)+'_0')(inputs)
    
    #Second layer
    x2 = keras.layers.Conv3D(filters=x2.shape[-1], kernel_size=3,padding="same",
                             activation="relu",name='incep-2_v2-3x3x3_'+str(n_layer)+'_1')(x2)
    x3 = keras.layers.Conv3D(filters=x3.shape[-1], kernel_size=5,padding="same",
                             activation="relu",name='incep-3_v2-5x5x5_'+str(n_layer)+'_1')(x3)
    x4 = keras.layers.Conv3D(filters=x4.shape[-1], kernel_size=1,padding="same",
                             activation="relu",name='incep-4_v2-1x1x1_'+str(n_layer)+'_1')(x4)
    
    return keras.layers.Concatenate(axis=-1, name="concat_incep_v2_"+str(n_layer))([x1,x2,x3,x4])

def inception_ours_naive(inputs, n_layer, output_filters):
    x1 = keras.layers.Conv3D(filters=output_filters//2, kernel_size=1,padding="same",
                            activation="relu",name="conv_channels-1_naive_"+str(n_layer))(inputs)
    x2 = keras.layers.Conv3D(filters=output_filters//2, kernel_size=1,padding="same",
                            activation="relu",name="conv_channels-2_naive_"+str(n_layer))(inputs)
    
    return keras.layers.Concatenate(axis=-1, name="concat_conv_channels_"+str(n_layer))([x1,x2])

def convTrans3D(inputs, n_layer, output_filters):
    return keras.layers.Conv3DTranspose(filters=output_filters, kernel_size=1, padding="same",
                                       activation="relu", name="conv3D_Trans_"+str(n_layer))(inputs)

def conv3D_Channels(inputs, n_layer, output_filters):
    return keras.layers.Conv3D(filters=output_filters, kernel_size=1, padding="same",
                              activation="relu", name="conv3D_channels_"+str(n_layer))(inputs)

def conv3D_asym(inputs, n_layer):
    """This method always return 1 filter in the output"""
    x = keras.layers.Reshape((inputs.shape[1],inputs.shape[2]*inputs.shape[3],inputs.shape[4]), 
                             name="reshape_"+str(n_layer))(inputs)
    x = keras.layers.Conv3D(filters=inputs.shape[-1]//4, kernel_size=(1,1,3), padding="same",
                           activation="relu", name="conv3D_simplified_"+str(n_layer))(x)
    return None