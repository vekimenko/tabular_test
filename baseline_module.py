import os
from typing import List
import tensorflow as tf
from tfx import v1 as tfx
import tensorflow_transform as tft
from tfx.components.trainer.fn_args_utils import DataAccessor
from tfx.components.trainer.fn_args_utils import FnArgs
from tfx_bsl.tfxio import dataset_options

# A list of the numerical feature names.

_LABEL_KEY = 'label'
NUMERIC_FEATURE_NAMES = [
    'avg_turnaround_time', 
    'avg_transfer_time', 
    'min_total_MB_consumed', 
    'max_total_MB_consumed', 
    'avg_total_MB_consumed', 
    'sum_total_MB_consumed', 
    'avg_cnt_request_id', 
    'cnt_total_MB_consumed', 
    'min_cnt_request_id', 
    'max_cnt_request_id', 
    'cnt_cnt_request_id', 
    'sum_cnt_request_id'
]
CATEGORICAL_FEATURE_NAMES = ['user_agent2','asn_country', 'asn_type', 'channel']
# WEIGHT_COLUMN_NAME = "fnlwgt"
FEATURE_NAMES = NUMERIC_FEATURE_NAMES + CATEGORICAL_FEATURE_NAMES
TARGET_FEATURE_NAME = "label"
TARGET_LABELS = ["legitimate", "non legitimate"]


LEARNING_RATE = 0.001
WEIGHT_DECAY = 0.0001
DROPOUT_RATE = 0.2
BATCH_SIZE = 265
NUM_EPOCHS = 5

NUM_TRANSFORMER_BLOCKS = 3  # Number of transformer blocks.
NUM_HEADS = 4  # Number of attention heads.
EMBEDDING_DIMS = 16  # Embedding dimensions of the categorical features.
MLP_HIDDEN_UNITS_FACTORS = [
    2,
    1,
]  # MLP hidden layer units, as factors of the number of inputs.
NUM_MLP_BLOCKS = 2  # Number of MLP blocks in the baseline model.


def preprocessing_fn(inputs):
    outputs = {}

    for key in NUMERIC_FEATURE_NAMES:
        outputs[key] = tft.scale_to_z_score(inputs[key])

    for key in CATEGORICAL_FEATURE_NAMES:
        outputs[key] = tft.compute_and_apply_vocabulary(
            inputs[key],
            top_k=10000,
            num_oov_buckets=1,
            vocab_filename=key
        )

    outputs[TARGET_FEATURE_NAME] = inputs[TARGET_FEATURE_NAME]
    
    for key in outputs:
        outputs[key] = tf.squeeze(outputs[key], -1)
    
    return outputs


def _apply_preprocessing(raw_features, tft_layer):
  transformed_features = tft_layer(raw_features)
  if TARGET_FEATURE_NAME in raw_features:
    transformed_label = transformed_features.pop(TARGET_FEATURE_NAME)
    return transformed_features, transformed_label
  else:
    return transformed_features, None


def _get_serve_examples_fn(model, tf_transform_output):
  model.tft_layer = tf_transform_output.transform_features_layer()

  signature_dict = {"avg_turnaround_time": tf.TensorSpec(shape=[], dtype=tf.float32,
                                                         name='avg_turnaround_time'),
                    "avg_transfer_time": tf.TensorSpec(shape=[None], dtype=tf.float32,
                                                       name='avg_transfer_time'),
                    "min_total_MB_consumed": tf.TensorSpec(shape=[None], dtype=tf.float32,
                                                           name='min_total_MB_consumed'),
                    "max_total_MB_consumed": tf.TensorSpec(shape=[None], dtype=tf.float32,
                                                           name='max_total_MB_consumed'),
                    "avg_total_MB_consumed": tf.TensorSpec(shape=[None], dtype=tf.float32,
                                                           name='avg_total_MB_consumed'),
                    "sum_total_MB_consumed": tf.TensorSpec(shape=[None], dtype=tf.int64,
                                                           name='sum_total_MB_consumed'),
                    "avg_cnt_request_id": tf.TensorSpec(shape=[None], dtype=tf.float32,
                                                        name='avg_cnt_request_id'),
                    "cnt_total_MB_consumed": tf.TensorSpec(shape=[None], dtype=tf.int64,
                                                           name='cnt_total_MB_consumed'),
                    "max_cnt_request_id": tf.TensorSpec(shape=[None], dtype=tf.float32,
                                                        name='max_cnt_request_id'),
                    "cnt_cnt_request_id": tf.TensorSpec(shape=[None], dtype=tf.int64,
                                                        name='cnt_cnt_request_id'),
                    "sum_cnt_request_id": tf.TensorSpec(shape=[None], dtype=tf.int64,
                                                        name='sum_cnt_request_id'),
                    "user_agent2": tf.TensorSpec(shape=[None], dtype=tf.string,
                                                 name='user_agent2'),
                    "asn_country": tf.TensorSpec(shape=[None], dtype=tf.string,
                                                 name='asn_country'),
                    "asn_type": tf.TensorSpec(shape=[None], dtype=tf.string,
                                              name='asn_type'),
                    "channel": tf.TensorSpec(shape=[None], dtype=tf.string,
                                             name='channel')
                    }

  @tf.function(input_signature=[signature_dict])
  def serve_examples_fn(serialized_tf_examples):
    # Expected input is a string which is serialized tf.Example format.
    # feature_spec = tf_transform_output.raw_feature_spec()
    # Because input schema includes unnecessary fields like 'species' and
    # 'island', we filter feature_spec to include required keys only.
    # required_feature_spec = {
    #     k: v for k, v in feature_spec.items() if k in _FEATURE_KEYS
    # }
    # parsed_features = tf.io.parse_example(serialized_tf_examples,
    #                                       required_feature_spec)
    parsed_features = serialized_tf_examples

    # Preprocess parsed input with transform operation defined in
    # preprocessing_fn().
    transformed_features, _ = _apply_preprocessing(parsed_features,
                                                   model.tft_layer)
    # Run inference with ML model.
    return model(transformed_features)

  return serve_examples_fn


def _input_fn(file_pattern: List[str],
              data_accessor: DataAccessor,
              tf_transform_output: tft.TFTransformOutput,
              is_train: bool = False,
              batch_size: int = BATCH_SIZE) -> tf.data.Dataset:
  dataset = data_accessor.tf_dataset_factory(
      file_pattern,
      dataset_options.TensorFlowDatasetOptions(
          batch_size=batch_size,
          label_key=TARGET_FEATURE_NAME),
      tf_transform_output.transformed_metadata.schema).repeat()

  return dataset


def create_model_inputs():
    inputs = {}
    for feature_name in FEATURE_NAMES:
        if feature_name in NUMERIC_FEATURE_NAMES:
            inputs[feature_name] = tf.keras.layers.Input(
                name=feature_name, shape=(1,), dtype=tf.float32
            )
        else:
            inputs[feature_name] = tf.keras.layers.Input(
                name=feature_name, shape=(1,), dtype=tf.int64
            )
    return inputs


def encode_inputs(inputs, embedding_dims, feature_vocab_sizes):
    encoded_categorical_feature_list = []
    numerical_feature_list = []

    for feature_name in inputs:

        if feature_name in CATEGORICAL_FEATURE_NAMES:
            vocab_size = feature_vocab_sizes[feature_name]
            embedding = tf.keras.layers.Embedding(
                input_dim=vocab_size + 1,
                output_dim=embedding_dims,
                name=f"{feature_name}_embedding")
            encoded_categorical_feature = embedding(inputs[feature_name])
            encoded_categorical_feature_list.append(encoded_categorical_feature)

        else:
            numerical_feature = tf.expand_dims(inputs[feature_name], -1)
            numerical_feature_list.append(numerical_feature)

    return encoded_categorical_feature_list, numerical_feature_list


def create_mlp(hidden_units, dropout_rate, activation, normalization_layer, name=None):
    mlp_layers = []
    for units in hidden_units:
        mlp_layers.append(normalization_layer()),
        mlp_layers.append(tf.keras.layers.Dense(units, activation=activation))
        mlp_layers.append(tf.keras.layers.Dropout(dropout_rate))

    return tf.keras.Sequential(mlp_layers, name=name)


def _build_keras_model(tft_output):

    feature_vocab_sizes = dict()
    for feature_name in CATEGORICAL_FEATURE_NAMES:
        feature_vocab_sizes[feature_name] = tft_output.vocabulary_size_by_name(
            feature_name
        )

    inputs = create_model_inputs()
    encoded_categorical_feature_list, numerical_feature_list = encode_inputs(
        inputs, EMBEDDING_DIMS, feature_vocab_sizes
    )
    features = tf.keras.layers.concatenate(
        encoded_categorical_feature_list + numerical_feature_list
    )
    feedforward_units = [features.shape[-1]]

    for layer_idx in range(NUM_MLP_BLOCKS):
        features = create_mlp(
            hidden_units=feedforward_units,
            dropout_rate=DROPOUT_RATE,
            activation=tf.keras.activations.gelu,
            normalization_layer=tf.keras.layers.LayerNormalization,
            name=f"feedforward_{layer_idx}",
        )(features)

    # Compute MLP hidden_units.
    mlp_hidden_units = [
        factor * features.shape[-1] for factor in MLP_HIDDEN_UNITS_FACTORS
    ]
    # Create final MLP.
    features = create_mlp(
        hidden_units=mlp_hidden_units,
        dropout_rate=DROPOUT_RATE,
        activation=tf.keras.activations.selu,
        normalization_layer=tf.keras.layers.BatchNormalization,
        name="MLP",
    )(features)

    # Add a sigmoid as a binary classifer.
    flatten = tf.keras.layers.Flatten()(features)
    outputs = tf.keras.layers.Dense(units=1, activation="sigmoid", name="sigmoid")(flatten)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model


def run_fn(fn_args: tfx.components.FnArgs):
  tf_transform_output = tft.TFTransformOutput(fn_args.transform_output)

  train_dataset = _input_fn(
      fn_args.train_files,
      fn_args.data_accessor,
      tf_transform_output,
      is_train=True,
      batch_size=BATCH_SIZE)

  eval_dataset = _input_fn(
      fn_args.eval_files,
      fn_args.data_accessor,
      tf_transform_output,
      is_train=False,
      batch_size=BATCH_SIZE)

  model = _build_keras_model(tf_transform_output)

  optimizer = tf.keras.optimizers.Adam(
      learning_rate=LEARNING_RATE, weight_decay=WEIGHT_DECAY
  )

  model.compile(
      optimizer=optimizer,
      loss=tf.keras.losses.BinaryCrossentropy(),
      metrics=[tf.keras.metrics.BinaryAccuracy(name="accuracy")],
  )

  model.fit(
      train_dataset,
      epochs=NUM_EPOCHS,
      steps_per_epoch=fn_args.train_steps,
      validation_data=eval_dataset,
      validation_steps=fn_args.eval_steps
  )

  signatures = {
      'serving_default': _get_serve_examples_fn(model, tf_transform_output),
  }
  model.save(fn_args.serving_model_dir, save_format='tf', signatures=signatures)




