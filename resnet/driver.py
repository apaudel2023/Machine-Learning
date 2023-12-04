from model_resnet import AutoEncoder
from gradcam import GradCAM
import pdb
import tensorflow as tf

input_shape = (64, 64, 1)  # Adjust the input shape based on your data
latent_dims = 10
model = AutoEncoder(input_shape=input_shape, latent_dim=latent_dims)


# model.summary()

# ae_model = tf.keras.models.Model(model.encoder, model.decoder)
# ae_model.summary()

# pdb.set_trace()
## 
xai = GradCAM(model)

