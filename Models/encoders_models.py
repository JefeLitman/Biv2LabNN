from tensorflow import keras

def reshape_layer(inputs, n_layer):
    return keras.layers.Reshape((inputs.shape[1],
                                 inputs.shape[2]*inputs.shape[3]*inputs.shape[4]),
                                 name="reshape_"+str(n_layer))(inputs)

def basic_encoder(inputs, cell_units, n_layer, with_sequences):
    x = reshape_layer(inputs, n_layer)
    return keras.layers.LSTM(units=cell_units, 
                             return_sequences=with_sequences, 
                             name="LSTM-"+str(cell_units)+"_"+str(n_layer))(x)

def unique_bidirectional_encoder(inputs, cell_units, n_layer, with_sequences):
    x = reshape_layer(inputs, n_layer)
    return keras.layers.Bidirectional(keras.layers.LSTM(units=cell_units,
                                                      return_sequences=with_sequences),
                                     merge_mode='concat',
                                     name="Bi-LSTM-"+str(cell_units)+"_"+str(n_layer)
                                    )(x)

def jota_encoder_slt(inputs, cell_units, n_layer, with_sequences):
    x = reshape_layer(inputs, n_layer)
    x, h1, c1 = keras.layers.LSTM(units = cell_units, return_sequences = True,
                            return_state=True,
                            name = "LTSM1-"+str(cell_units)+"_"+str(n_layer))(x)
    x = keras.layers.LSTM(units = cell_units, return_sequences = with_sequences,
                            name = "LTSM2-"+str(cell_units)+"_"+str(n_layer))(x, initial_state=[h1,c1])
    return x