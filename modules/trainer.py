import os
import tensorflow as tf
import tensorflow_transform as tft
import tensorflow_hub

from tfx.components.trainer.fn_args_utils import FnArgs

LABEL_KEY = "labels"
FEATURE_KEY = "text"
VOCAB_SIZE = 10000
SEQUENCE_LENGTH = 100
EMBEDDING_DIM = 16

vector_layer = tf.keras.layers.TextVectorization(
    max_tokens=VOCAB_SIZE,
    output_sequence_length=SEQUENCE_LENGTH
)


def gzip_reader_fn(filenames):
    return tf.data.TFRecordDataset(filenames, compression_type='GZIP')

def input_fn(file_pattern, 
             tf_transform_output,
             num_epochs:int,
             batch_size:int=128,
            ) -> tf.data.Dataset:
    
    transformed_feature_spec = (
        tf_transform_output.transformed_feature_spec().copy()
    )
    
    dataset = tf.data.experimental.make_batched_features_dataset(
        file_pattern = file_pattern,
        batch_size = batch_size,
        features = transformed_feature_spec,
        reader = gzip_reader_fn,
        num_epochs = num_epochs,
        label_key = str(LABEL_KEY + "_xf")
    )
    
    return dataset

def model_fn() -> tf.keras.Model:
    inputs = tf.keras.Input(shape=(1,), name=str(FEATURE_KEY + "_xf"), dtype=tf.string)
    reshape_layer = tf.reshape(inputs, [-1])
    x = vector_layer(reshape_layer)
    x = tf.keras.layers.Embedding(VOCAB_SIZE, EMBEDDING_DIM, name="embedding")(x)
    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    x = tf.keras.layers.Dense(64, activation='relu')(x)
    x = tf.keras.layers.Dense(32, activation="relu")(x)
    x = tf.keras.layers.Dense(16, activation='relu')(x)
    outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)
    model = tf.keras.Model(inputs=inputs, outputs = outputs)
    
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    model.summary()
    return model

def _get_serve_tf_examples_fn(model, tf_transform_output):
    model.tft_layer = tf_transform_output.transform_features_layer()
    
    @tf.function
    def serve_tf_examples_fn(serialized_tf_examples):
        feature_spec = tf_transform_output.raw_feature_spec()
        feature_spec.pop(LABEL_KEY)
        parsed_features = tf.io.parse_example(
            serialized_tf_examples,
            feature_spec
        )
        
        transformed_features = model.tft_layer(parsed_features)
        return model(transformed_features)
    
    return serve_tf_examples_fn

def run_fn(fn_args: FnArgs):
    early_stopping = tf.keras.callbacks.EarlyStopping(
        patience=5,
        monitor='accuracy',
        mode='max',
        verbose=1
    )
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        fn_args.serving_model_dir,
        monitor='accuracy',
        mode='max',
        verbose=1,
        save_best_only=True
    )

    tensorflow_transform_output = tft.TFTransformOutput(fn_args.transform_output)

    train_dataset = input_fn(
        fn_args.train_files,
        tf_transform_output=tensorflow_transform_output,
        num_epochs=10
    )
    validation_dataset = input_fn(
        fn_args.eval_files,
        tf_transform_output=tensorflow_transform_output,
        num_epochs=10
    )

    vector_layer.adapt(
        [
        j[0].numpy()[0]
        for j in [data[0][str(FEATURE_KEY + "_xf")] 
        for data in list(train_dataset)
        ]
        ]
    )
    model = model_fn()
    model.fit(
        train_dataset,
        validation_data=validation_dataset,
        epochs=10,
        callbacks=[early_stopping, checkpoint]
    )
    signatures = {
        'serving_default': _get_serve_tf_examples_fn(
             model,
             tensorflow_transform_output).get_concrete_function(
                 tf.TensorSpec(
                     shape=[None],
                     dtype=tf.string,
                     name='examples')
        )
    }
    model.save(
        fn_args.serving_model_dir,
        save_format='tf',
        signatures=signatures
    )



