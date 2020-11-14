# Keras implementation of the paper:
# 3D MRI Brain Tumor Segmentation Using Autoencoder Regularization
# by Myronenko A. (https://arxiv.org/pdf/1810.11654.pdf)
# Author of this code: Suyog Jadhav (https://github.com/IAmSUyogJadhav)

### >>> For GenomeDB server
# from tensorflow.compat.v1 import ConfigProto
# from tensorflow.compat.v1 import InteractiveSession

# config = ConfigProto()
# config.gpu_options.allow_growth = True
# session = InteractiveSession(config=config)
### <<< 

import tensorflow.keras.backend as K
from tensorflow.keras.losses import mse
from tensorflow.keras.layers import Conv3D, Activation, Add, UpSampling3D, Lambda, Dense
from tensorflow.keras.layers import Input, Reshape, Flatten, Dropout, SpatialDropout3D
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
# from tensorflow.keras.utils import multi_gpu_model
import tensorflow_addons as tfa
from tensorflow_addons.layers import GroupNormalization
from tensorflow import keras
import tensorflow as tf


class green_block(keras.layers.Layer):
    def __init__(self, filters, regularizer, data_format='channels_first', name=None, **kwargs):
        super().__init__(**kwargs)
        self.hidden = [
            Conv3D(filters=filters, kernel_size=(1, 1, 1), strides=1, kernel_regularizer = regularizer, data_format=data_format, name=f'Res_{name}' if name else None),
            GroupNormalization(groups = 8, axis = 1 if data_format == 'channels_first' else 0, name = f'GroupNorm_1_{name}' if name else None),
            Activation('relu', name=f'Relu_1_{name}' if name else None),
            Conv3D(filters=filters, kernel_size=(3, 3, 3), strides=1, padding='same', kernel_regularizer = regularizer, data_format=data_format, name=f'Conv3D_1_{name}' if name else None),
            GroupNormalization(groups = 8, axis = 1 if data_format == 'channels_first' else 0, name = f'GroupNorm_2_{name}' if name else None),
            Activation('relu', name=f'Relu_2_{name}' if name else None),
            Conv3D(filters=filters, kernel_size=(3, 3, 3), strides=1, padding='same', kernel_regularizer = regularizer, data_format=data_format, name=f'Conv3D_2_{name}' if name else None),
            Add(name=f'Out_{name}' if name else None)
        ]
    
    def call(self, inputs):
        Z = inputs
        inp_res = self.hidden[0](Z)
        Z = self.hidden[1](Z)
        
        for layer in self.hidden[2:7]:
            Z = layer(Z)
            
        Z = self.hidden[7]([Z, inp_res])
        
        return Z


# From keras-team/keras/blob/master/examples/variational_autoencoder.py
class sampling(keras.layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


def dice_coefficient(y_true, y_pred):
    intersection = K.sum(K.abs(y_true * y_pred), axis=[-3,-2,-1])
    dn = K.sum(K.square(y_true) + K.square(y_pred), axis=[-3,-2,-1]) + 1e-8
    return K.mean(2 * intersection / dn, axis=[0,1])


class DiceLoss(keras.losses.Loss):
    def call(self, y_true, y_pred):
        y_pred = tf.convert_to_tensor(y_pred)
        y_true = tf.cast(y_true, y_pred.dtype)
        
        intersection = tf.reduce_sum(tf.abs(y_true * y_pred), axis=[-3,-2,-1])
        dn = tf.reduce_sum(tf.square(y_true) + tf.square(y_pred), axis=[-3,-2,-1]) + 1e-8
        return - tf.reduce_mean(2 * intersection / dn, axis=-1)

    
class LossVAE(keras.layers.Layer):
    def __init__(self, weight_L2, weight_KL, n, **kwargs):
        super().__init__(**kwargs)
        
        self.weight_L2 = weight_L2
        self.weight_KL = weight_KL
        self.n = n
        
    def call(self, inputs):
        x, out_VAE, z_mean, z_var = inputs
        loss_L2 = tf.reduce_mean(tf.square(x - out_VAE), axis=(1, 2, 3, 4)) # original axis value is (1,2,3,4).
        loss_KL = (1 / self.n) * tf.reduce_sum(
            tf.exp(z_var) + tf.square(z_mean) - 1. - z_var,
            axis=-1
        )
        
        VAE_loss = tf.reduce_mean(tf.add(self.weight_L2 * loss_L2, self.weight_KL * loss_KL, name = "add_L2_KL"), name = "mean_VAELoss")
        self.add_loss(VAE_loss)
        
        return         
    
    
    
def loss_gt(e=1e-8):
    """
    loss_gt(e=1e-8)
    ------------------------------------------------------
    Since keras does not allow custom loss functions to have arguments
    other than the true and predicted labels, this function acts as a wrapper
    that allows us to implement the custom loss used in the paper. This function
    only calculates - L<dice> term of the following equation. (i.e. GT Decoder part loss)
    
    L = - L<dice> + weight_L2 ∗ L<L2> + weight_KL ∗ L<KL>
    
    Parameters
    ----------
    `e`: Float, optional
        A small epsilon term to add in the denominator to avoid dividing by
        zero and possible gradient explosion.
        
    Returns
    -------
    loss_gt_(y_true, y_pred): A custom keras loss function
        This function takes as input the predicted and ground labels, uses them
        to calculate the dice loss.
        
    """
    def loss_gt_(y_true, y_pred):
        intersection = K.sum(K.abs(y_true * y_pred), axis=[-3,-2,-1])
        dn = K.sum(K.square(y_true) + K.square(y_pred), axis=[-3,-2,-1]) + e
        
        return - K.mean(2 * intersection / dn, axis=[0,1])
    
    return loss_gt_

def loss_VAE(input_shape, z_mean, z_var, weight_L2=0.1, weight_KL=0.1):
    """
    loss_VAE(input_shape, z_mean, z_var, weight_L2=0.1, weight_KL=0.1)
    ------------------------------------------------------
    Since keras does not allow custom loss functions to have arguments
    other than the true and predicted labels, this function acts as a wrapper
    that allows us to implement the custom loss used in the paper. This function
    calculates the following equation, except for -L<dice> term. (i.e. VAE decoder part loss)
    
    L = - L<dice> + weight_L2 ∗ L<L2> + weight_KL ∗ L<KL>
    
    Parameters
    ----------
     `input_shape`: A 4-tuple, required
        The shape of an image as the tuple (c, H, W, D), where c is
        the no. of channels; H, W and D is the height, width and depth of the
        input image, respectively.
    `z_mean`: An keras.layers.Layer instance, required
        The vector representing values of mean for the learned distribution
        in the VAE part. Used internally.
    `z_var`: An keras.layers.Layer instance, required
        The vector representing values of variance for the learned distribution
        in the VAE part. Used internally.
    `weight_L2`: A real number, optional
        The weight to be given to the L2 loss term in the loss function. Adjust to get best
        results for your task. Defaults to 0.1.
    `weight_KL`: A real number, optional
        The weight to be given to the KL loss term in the loss function. Adjust to get best
        results for your task. Defaults to 0.1.
        
    Returns
    -------
    loss_VAE_(y_true, y_pred): A custom keras loss function
        This function takes as input the predicted and ground labels, uses them
        to calculate the L2 and KL loss.
        
    """
    def loss_VAE_(y_true, y_pred):
        c, H, W, D = input_shape
        n = c * H * W * D
        
        loss_L2 = K.mean(K.square(y_true - y_pred), axis=(1, 2, 3, 4)) # original axis value is (1,2,3,4).

        loss_KL = (1 / n) * K.sum(
            K.exp(z_var) + K.square(z_mean) - 1. - z_var,
            axis=-1
        )

        return loss_L2 + loss_KL

    return loss_VAE_


class conv3d_autoenc_reg(keras.Model):
    def __init__(self, input_shape=(4, 160, 192, 128), output_channels=3, l2_reg_weight = 1e-5, weight_L2=0.1, weight_KL=0.1, 
                 dice_e=1e-8, test_mode = True, n_gpu = 1, GL_weight = 1, VL_weight = 0.1, **kwargs):
        super().__init__(**kwargs)

        self.c, self.H, self.W, self.D = input_shape
        self.n = self.c * self.H * self.W * self.D
        assert len(input_shape) == 4, "Input shape must be a 4-tuple"
        if test_mode is not True: assert (self.c % 4) == 0, "The no. of channels must be divisible by 4"
        assert (self.H % 16) == 0 and (self.W % 16) == 0 and (self.D % 16) == 0, "All the input dimensions must be divisible by 16"
        self.l2_regularizer = l2(l2_reg_weight) if l2_reg_weight is not None else None
        
        self.input_shape_p = input_shape
        self.output_channels = output_channels
        self.l2_reg_weight = l2_reg_weight
        self.weight_L2 = weight_L2
        self.weight_KL = weight_KL
        self.dice_e = dice_e
        self.GL_weight = GL_weight
        self.VL_weight = VL_weight
        
        self.LossVAE = LossVAE(weight_L2, weight_KL, self.n)
        
        ## The Initial Block
        self.Input_x1 = Conv3D(
        filters=32,
        kernel_size=(3, 3, 3),
        strides=1,
        padding='same',
        kernel_regularizer = self.l2_regularizer,
        data_format='channels_first',
        name='Input_x1')
        
        ## Dropout (0.2)
        self.spatial_dropout = SpatialDropout3D(0.2, data_format='channels_first')
        
        ## Green Block x1 (output filters = 32)
        self.x1 = green_block(32, regularizer = self.l2_regularizer, name='x1')
        self.Enc_DownSample_32 = Conv3D(
            filters=32,
            kernel_size=(3, 3, 3),
            strides=2,
            padding='same',
            kernel_regularizer = self.l2_regularizer,
            data_format='channels_first',
            name='Enc_DownSample_32')
        
        ## Green Block x2 (output filters = 64)
        self.Enc_64_1 = green_block(64, regularizer = self.l2_regularizer, name='Enc_64_1')
        self.x2 = green_block(64, regularizer = self.l2_regularizer, name='x2')
        self.Enc_DownSample_64 = Conv3D(
                            filters=64,
                            kernel_size=(3, 3, 3),
                            strides=2,
                            padding='same',
                            kernel_regularizer = self.l2_regularizer,
                            data_format='channels_first',
                            name='Enc_DownSample_64')
        
        ## Green Blocks x2 (output filters = 128)
        self.Enc_128_1 = green_block(128, regularizer = self.l2_regularizer, name='Enc_128_1')
        self.x3 = green_block(128, regularizer = self.l2_regularizer, name='x3')
        self.Enc_DownSample_128 = Conv3D(filters=128, kernel_size=(3, 3, 3), strides=2, padding='same', kernel_regularizer = self.l2_regularizer, 
                                         data_format='channels_first', name='Enc_DownSample_128')
        
        ## Green Blocks x4 (output filters = 256)
        self.Enc_256_1 = green_block(256, regularizer = self.l2_regularizer, name='Enc_256_1')
        self.Enc_256_2 = green_block(256, regularizer = self.l2_regularizer, name='Enc_256_2')
        self.Enc_256_3 = green_block(256, regularizer = self.l2_regularizer, name='Enc_256_3')
        self.x4 = green_block(256, regularizer = self.l2_regularizer, name='x4')
        
        # -------------------------------------------------------------------------
        # Decoder
        # -------------------------------------------------------------------------

        ## GT (Groud Truth) Part
        # -------------------------------------------------------------------------
        
        ### Green Block x1 (output filters=128)
        self.Dec_GT_ReduceDepth_128 = Conv3D(filters=128, kernel_size=(1, 1, 1), strides=1, kernel_regularizer = self.l2_regularizer, data_format='channels_first', name='Dec_GT_ReduceDepth_128')
        self.Dec_GT_UpSample_128 = UpSampling3D(size=2, data_format='channels_first', name='Dec_GT_UpSample_128') 
        self.Input_Dec_GT_128 = Add(name='Input_Dec_GT_128')
        self.Dec_GT_128 = green_block(128, regularizer = self.l2_regularizer, name='Dec_GT_128')
        
        ### Green Block x1 (output filters=64)
        self.Dec_GT_ReduceDepth_64 = Conv3D(filters=64, kernel_size=(1, 1, 1), strides=1, kernel_regularizer = self.l2_regularizer, data_format='channels_first', name='Dec_GT_ReduceDepth_64')
        self.Dec_GT_UpSample_64 = UpSampling3D(size=2, data_format='channels_first', name='Dec_GT_UpSample_64')
        self.Input_Dec_GT_64 = Add(name='Input_Dec_GT_64')
        self.Dec_GT_64 = green_block(64, regularizer = self.l2_regularizer, name='Dec_GT_64')
        
        ### Green Block x1 (output filters=32)
        self.Dec_GT_ReduceDepth_32 = Conv3D(filters=32, kernel_size=(1, 1, 1), strides=1, kernel_regularizer = self.l2_regularizer, data_format='channels_first', 
                                       name='Dec_GT_ReduceDepth_32')
        self.Dec_GT_UpSample_32 = UpSampling3D(size=2, data_format='channels_first', name='Dec_GT_UpSample_32')
        self.Input_Dec_GT_32 = Add(name='Input_Dec_GT_32')
        self.Dec_GT_32 = green_block(32, regularizer = self.l2_regularizer, name='Dec_GT_32')
        
        ### Blue Block x1 (output filters=32)
        self.Input_Dec_GT_Output = Conv3D(filters=32, kernel_size=(3, 3, 3), strides=1, padding='same', kernel_regularizer = self.l2_regularizer, 
                                     data_format='channels_first', name='Input_Dec_GT_Output')
        
        ### Output Block
        self.Dec_GT_Output = Conv3D(filters=self.output_channels, kernel_size=(1, 1, 1), strides=1, kernel_regularizer = self.l2_regularizer, 
                                data_format='channels_first', activation='sigmoid', name='Dec_GT_Output')
        
        ## VAE (Variational Auto Encoder) Part
        # -------------------------------------------------------------------------

        ### VD Block (Reducing dimensionality of the data)
        self.Dec_VAE_VD_GN = GroupNormalization(groups=8, axis=1, name='Dec_VAE_VD_GN')
        self.Dec_VAE_VD_relu = Activation('relu', name='Dec_VAE_VD_relu')
        self.Dec_VAE_VD_Conv3D = Conv3D(filters=16, kernel_size=(3, 3, 3), strides=2, padding='same', kernel_regularizer = self.l2_regularizer, 
                                   data_format='channels_first', name='Dec_VAE_VD_Conv3D')
        
        # Not mentioned in the paper, but the author used a Flattening layer here.
        self.Dec_VAE_VD_Flatten = Flatten(name='Dec_VAE_VD_Flatten')
        self.Dec_VAE_VD_Dense = Dense(256, name='Dec_VAE_VD_Dense')

        ### VDraw Block (Sampling)
        self.Dec_VAE_VDraw_Mean = Dense(128, name='Dec_VAE_VDraw_Mean')
        self.Dec_VAE_VDraw_Var = Dense(128, name='Dec_VAE_VDraw_Var')
#         self.Dec_VAE_VDraw_Sampling = Lambda(sampling, name='Dec_VAE_VDraw_Sampling')
        self.Dec_VAE_VDraw_Sampling = sampling()

        ### VU Block (Upsizing back to a depth of 256)
        c1 = 1
        self.VU_Dense1 = Dense((c1) * (self.H//16) * (self.W//16) * (self.D//16))
        self.VU_relu = Activation('relu')
        self.VU_reshape = Reshape(((c1), (self.H//16), (self.W//16), (self.D//16)))
        self.Dec_VAE_ReduceDepth_256 = Conv3D(filters=256, kernel_size=(1, 1, 1), strides=1, kernel_regularizer = self.l2_regularizer, data_format='channels_first',
                                            name='Dec_VAE_ReduceDepth_256')
        self.Dec_VAE_UpSample_256 = UpSampling3D(size=2, data_format='channels_first', name='Dec_VAE_UpSample_256')

        ### Green Block x1 (output filters=128)
        self.Dec_VAE_ReduceDepth_128 = Conv3D(filters=128, kernel_size=(1, 1, 1), strides=1, kernel_regularizer = self.l2_regularizer, data_format='channels_first', 
                                         name='Dec_VAE_ReduceDepth_128')
        self.Dec_VAE_UpSample_128 = UpSampling3D(size=2, data_format='channels_first', name='Dec_VAE_UpSample_128')
        self.Dec_VAE_128 = green_block(128, regularizer = self.l2_regularizer, name='Dec_VAE_128')

        ### Green Block x1 (output filters=64)
        self.Dec_VAE_ReduceDepth_64 = Conv3D(filters=64, kernel_size=(1, 1, 1), strides=1, kernel_regularizer = self.l2_regularizer, data_format='channels_first',
                                        name='Dec_VAE_ReduceDepth_64')
        self.Dec_VAE_UpSample_64 = UpSampling3D(size=2, data_format='channels_first', name='Dec_VAE_UpSample_64')
        self.Dec_VAE_64 = green_block(64, regularizer = self.l2_regularizer, name='Dec_VAE_64')

        ### Green Block x1 (output filters=32)
        self.Dec_VAE_ReduceDepth_32 = Conv3D(filters=32, kernel_size=(1, 1, 1), strides=1, kernel_regularizer = self.l2_regularizer, data_format='channels_first',
                                        name='Dec_VAE_ReduceDepth_32')
        self.Dec_VAE_UpSample_32 = UpSampling3D(size=2, data_format='channels_first', name='Dec_VAE_UpSample_32')
        self.Dec_VAE_32 = green_block(32, regularizer = self.l2_regularizer, name='Dec_VAE_32')

        ### Blue Block x1 (output filters=32)
        self.Input_Dec_VAE_Output = Conv3D(filters=32, kernel_size=(3, 3, 3), strides=1, padding='same', kernel_regularizer = self.l2_regularizer, 
                                      data_format='channels_first', name='Input_Dec_VAE_Output')

        ### Output Block
        self.Dec_VAE_Output = Conv3D(filters=self.c, kernel_size=(1, 1, 1), strides=1, kernel_regularizer = self.l2_regularizer, data_format='channels_first', 
                                     name='Dec_VAE_Output')
        
#     def build(self, batch_input_shape):
#         n_inputs = batch_input_shape[-1]
        
#         ### super build
#         super().build(batch_input_shape)
        
    def call(self, inputs, training=None):
        Z = inputs
        x = self.Input_x1(Z)
        
        ## Dropout (0.2)
        x = self.spatial_dropout(x)

        ## Green Block x1 (output filters = 32)
        x1 = self.x1(x)
        x = self.Enc_DownSample_32(x1)

        ## Green Block x2 (output filters = 64)
        x = self.Enc_64_1(x)
        x2 = self.x2(x)
        x = self.Enc_DownSample_64(x2)

        ## Green Blocks x2 (output filters = 128)
        x = self.Enc_128_1(x)
        x3 = self.x3(x)
        x = self.Enc_DownSample_128(x3)

        ## Green Blocks x4 (output filters = 256)
        x = self.Enc_256_1(x)
        x = self.Enc_256_2(x)
        x = self.Enc_256_3(x)
        x4 = self.x4(x)

        # -------------------------------------------------------------------------
        # Decoder
        # -------------------------------------------------------------------------

        ## GT (Groud Truth) Part
        # -------------------------------------------------------------------------

        ### Green Block x1 (output filters=128)
        x = self.Dec_GT_ReduceDepth_128(x4)
        x = self.Dec_GT_UpSample_128(x)
        x = self.Input_Dec_GT_128([x, x3])
        x = self.Dec_GT_128(x)

        ### Green Block x1 (output filters=64)
        x = self.Dec_GT_ReduceDepth_64(x)
        x = self.Dec_GT_UpSample_64(x)
        x = self.Input_Dec_GT_64([x, x2])
        x = self.Dec_GT_64(x)

        ### Green Block x1 (output filters=32)
        x = self.Dec_GT_ReduceDepth_32(x)
        x = self.Dec_GT_UpSample_32(x)
        x = self.Input_Dec_GT_32([x, x1])
        x = self.Dec_GT_32(x)

        ### Blue Block x1 (output filters=32)
        x = self.Input_Dec_GT_Output(x)

        ### Output Block
        out_GT = self.Dec_GT_Output(x)

        ## VAE (Variational Auto Encoder) Part
        # -------------------------------------------------------------------------

        ### VD Block (Reducing dimensionality of the data)
        x = self.Dec_VAE_VD_GN(x4)
        x = self.Dec_VAE_VD_relu(x)
        x = self.Dec_VAE_VD_Conv3D(x)

        # Not mentioned in the paper, but the author used a Flattening layer here.
        x = self.Dec_VAE_VD_Flatten(x)
        x = self.Dec_VAE_VD_Dense(x)

        ### VDraw Block (Sampling)
        z_mean = self.Dec_VAE_VDraw_Mean(x)
        z_var = self.Dec_VAE_VDraw_Var(x)
        x = self.Dec_VAE_VDraw_Sampling([z_mean, z_var])

        ### VU Block (Upsizing back to a depth of 256)
        x = self.VU_Dense1(x)
        x = self.VU_relu(x)
        x = self.VU_reshape(x)
        x = self.Dec_VAE_ReduceDepth_256(x)
        x = self.Dec_VAE_UpSample_256(x)

        ### Green Block x1 (output filters=128)
        x = self.Dec_VAE_ReduceDepth_128(x)
        x = self.Dec_VAE_UpSample_128(x)
        x = self.Dec_VAE_128(x)

        ### Green Block x1 (output filters=64)
        x = self.Dec_VAE_ReduceDepth_64(x)
        x = self.Dec_VAE_UpSample_64(x)
        x = self.Dec_VAE_64(x)

        ### Green Block x1 (output filters=32)
        x = self.Dec_VAE_ReduceDepth_32(x)
        x = self.Dec_VAE_UpSample_32(x)
        x = self.Dec_VAE_32(x)

        ### Blue Block x1 (output filters=32)
        x = self.Input_Dec_VAE_Output(x)

        ### Output Block
        out_VAE = self.Dec_VAE_Output(x) 
        
        self.LossVAE([Z, out_VAE, z_mean, z_var])
        
        return out_GT