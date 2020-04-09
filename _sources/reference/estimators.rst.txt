Estimator Classes
=================
All estimators inherit from the following abstract base class:

.. toctree::
   :maxdepth: 2

	EstimatorBase <estimators/estimator_base>

Deep Learning
"""""""""""""

The following deep learning models utilize TensorFlow with the Keras high-level
interface. All current implementations inherit from the following abstract base
class:

.. toctree::
   :maxdepth: 2

   TFEstimatorBase <estimators/tf_estimator_base>

Current deep learning model implementations:

.. toctree::
   :maxdepth: 2

   Conv6 <estimators/conv6>
   HwyBLSTM (LSTM++) <estimators/hwy_blstm>
   LSTM <estimators/lstm>
   MLP <estimators/mlp>

Other deep learning classes:

.. toctree::
   :maxdepth: 2

   TFEpochLogger <estimators/tf_epoch_logger>
   HighwayLayer <estimators/highway_layer>

Evolutionary
""""""""""""

Current evolutionary based estimators:

.. toctree::
   :maxdepth: 2

   BasicGA <estimators/basic_ga>
   NSGA3 <estimators/nsga3>
