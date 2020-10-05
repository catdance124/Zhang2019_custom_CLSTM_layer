# Zhang2019_custom_CLSTM
This repo includes Zhang2019's CLSTM implemented using keras(tensorflow2).  
```
Zhang, Haokui, et al. "Exploiting temporal consistency for real-time video depth estimation."
Proceedings of the IEEE International Conference on Computer Vision. 2019.
```

Author's implimentation using pytorch is here.  
https://github.com/hkzhang91/ST-CLSTM/blob/master/CLSTM_Depth_Estimation-master/models_CLTSM/R_CLSTM_modules_2.py

## Implementation environment
```
OS Windows10
CUDA Toolkit 10.1 update2
cuDNN v7.6.5 (November 5th, 2019), for CUDA 10.1
Python 3.6.6 (anaconda3)
tensorflow 2.3.0
keras 2.4.3
```

## File structure
```
Zhang2019_custom_CLSTM/
  ┣━━ README.md    ...    this doc.
  ┣━━ convolutional_recurrent.py ... for calling keras's ConvRNN2D locally
  ┗━━ STConvLSTM2DCell.py    ...    main CLSTM cell
```
**Note** that convolutional_recurrent.py is wget from keras [v2.3.0](https://github.com/tensorflow/tensorflow/blob/v2.3.0/tensorflow/python/keras/layers/convolutional_recurrent.py).
```sh
$ wget https://raw.githubusercontent.com/tensorflow/tensorflow/v2.3.0/tensorflow/python/keras/layers/convolutional_recurrent.py
```


# How to use
git clone
```sh
$ git clone https://github.com/catdance124/Zhang2019_custom_CLSTM.git
```
The following is an example of how to call a custom CLSTM.
```python
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from convolutional_recurrent import ConvRNN2D
from STConvLSTM2DCell import STConvLSTM2DCell

inputs = Input((None, 256, 256, 3))
x = ConvRNN2D(STConvLSTM2DCell(8, kernel_size=3, padding='same', 
                    activation='tanh', recurrent_activation='hard_sigmoid',
                    kernel_initializer='glorot_uniform', recurrent_initializer='orthogonal'), 
            return_sequences=True, name='STConvLSTM2D')(inputs)
model = Model(inputs=inputs, outputs=x)
model.summary()

# _________________________________________________________________
# Layer (type)                 Output Shape              Param #   
# =================================================================
# input_2 (InputLayer)         [(None, None, 256, 256, 3 0         
# _________________________________________________________________
# STConvLSTM2D (ConvRNN2D)     (None, None, 256, 256, 1) 9914      
# =================================================================
# Total params: 9,914
# Trainable params: 9,870
# Non-trainable params: 44
# _________________________________________________________________
```

# Implementation description
In Japanese: 好きな構造のconvlutional RNNを組み立てる(tensorflow2/keras)  
https://catdance124.hatenablog.jp/entry/2020/10/04/211805