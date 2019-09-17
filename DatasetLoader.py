"""UCF & HMDB dataset input module.
"""

from random import randint
import pathlib
import tensorflow as tf

def seleccionar_cuadro_aleatorio(imagen, nueva_dimension):
    """Selecciona un cuadro aleatorio de dimension nueva_dimensionxnueva_dimension en la imagen"""
    pos_y = randint(0,imagen.shape[0].value - nueva_dimension)
    pos_x = randint(0,imagen.shape[1].value - nueva_dimension)
    return imagen[pos_y : pos_y + nueva_dimension , pos_x : pos_x + nueva_dimension, :]

def seleccionar_extension_temporal(video, nro_frames):
    """Selecciona la cantidad nro_frames sobre el video de forma aleatoria y lo retorna"""
    extension = randint(0,video.shape[0].value - nro_frames)
    return video[extension : extension + nro_frames, : , :, :]

def get_entrada(tipo_dataset, data_path, data_type, len_frames, batch_size, modo):
    """Construye e inicializa el dataset especificado.

    Argumentos:
      tipo_dataset: Si es 'ufc101' o 'hmdb51'.
      data_path: Ruta donde estan todos los archivos.
      data_type: Si solamente es 'rgb' o 'flow' si vienen con flujo.
      len_frames: La cantidad de frames del video.
      batch_size: Tamaño de los batch.
      modo: Si va ser 'train' o 'test'.
    Returns:
      entrada: El tensor de entrada de la forma [batch_size, len_frames, tam_imagen, tam_imagen, nro_canales]
      labels: El tensor de las etiquetas de las entradas [batch_size, num_clases]
    Raises:
      ValueError: when the specified dataset is not supported.
    """

    root_path = pathlib.Path("../DataSets/HMDB51")
    rgb_videos_path = list(root_path.glob('frames/*'))
    rgb_videos_path = [str(video) for video in rgb_videos_path]

    """El uso del flujo aun no esta habilitado"""
    # flow_videos_path = list(root_path.glob('frames/*'))
    # flow_videos_path_path = [str(video) for video in flow_videos_path]

    if tipo_dataset == 'ucf101':
        num_classes = 101

        """Extraccion de los nombres de las clases para este dataset"""
        nombres_clases = sorted(pathlib.Path(item).name for item in rgb_videos_path if pathlib.Path(item).is_dir())
        nombres_clases = [clase.split("_")[1] for clase in nombres_clases]

        # Extraccion de todos los tipos de clases en un vector de python
        clases = []
        for clase in nombres_clases:
            if clase not in clases:
                clases.append(clase)

        nombres_clases = clases

        clase_a_numero = dict((name, index) for index, name in enumerate(nombres_clases))

        """Clases videos es el conjunto de numero de clases para cada video del dataset"""
        clases_videos = [clase_a_numero[pathlib.Path(item).name.split("_")[1]] for item in rgb_videos_path]

    elif tipo_dataset == 'hmdb51':
        num_classes = 51

    else:
        raise ValueError('Not supported dataset %s', tipo_dataset)

    if modo == 'train':

    else:



    assert len(images.get_shape()) == 4
    assert images.get_shape()[0] == batch_size
    assert images.get_shape()[-1] == 3
    assert len(labels.get_shape()) == 2
    assert labels.get_shape()[0] == batch_size
    assert labels.get_shape()[1] == num_classes

    # Display the training images in the visualizer.
    tf.summary.image('images', images)
    return images, labels