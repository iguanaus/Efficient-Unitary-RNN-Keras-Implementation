# Efficient-Unitary-RNN Keras Implementation
This is a keras version of the EURNN that allows new models to be constructed by the line:
  - model.add(EURNNCell(80,input_shape=(history,d),return_sequences='true'))

##Setup:
Note that the keras setup file must be switched to use tensorflow as the backend.


##Notes:

The output dimension is the same as the number of units.
The architecture is simply:
  hidden = activation(Ux+Wh,bias)
  where the hidden is thus the output. 


For original EUNN and EURNN code, see [`EUNN-tensorflow`](https://github.com/jingli9111/EUNN-tensorflow)

For more detail, see [paper](https://arxiv.org/pdf/1612.05231.pdf)
