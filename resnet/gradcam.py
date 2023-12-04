import tensorflow as tf
import pdb


class GradCAM:

    def __init__(self, model):
        # self.model = model
        model.encoder.summary()
        print(f'total layers in model:{len(model.encoder.layers)}')

        encoder_model = model.encoder
        # Find the index of the last convolutional layer in the encoder model
        last_conv_layer_index = None
        for i, layer in enumerate(encoder_model.layers[::-1]):
            if isinstance(layer, tf.keras.layers.Conv2D):
                last_conv_layer_index = i
                break

        if last_conv_layer_index is not None:
            last_conv_layer_index = len(encoder_model.layers) - last_conv_layer_index - 1

        # Build the first submodel up to the last convolutional layer
        submodel_1 = tf.keras.models.Model(inputs=encoder_model.input, outputs=encoder_model.layers[last_conv_layer_index].output, name="SubModel1")

        # Build the second submodel from the output of the first submodel to the end
        submodel_2_layers = encoder_model.layers[last_conv_layer_index + 1:]
        submodel_2_input = tf.keras.layers.Input(shape=submodel_1.output.shape[1:])
        y = [submodel_2_input]
        for layer in submodel_2_layers:
            y = layer(y)
        submodel_2_output = y
        submodel_2 = tf.keras.models.Model(inputs=submodel_2_input, outputs=submodel_2_output, name="SubModel2")

        # Summary of the submodels
        submodel_1.summary()
        submodel_2.summary()



    

    

