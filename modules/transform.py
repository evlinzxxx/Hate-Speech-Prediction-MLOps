import tensorflow as tf

LABEL_KEY = "labels"
FEATURE_KEY = "text"

def preprocessing_fn(inputs):
    print("DEBUG: Data masuk ke Transform:", inputs.keys())

    if FEATURE_KEY not in inputs:
        raise KeyError(f"ERROR: Kolom {FEATURE_KEY} tidak ditemukan! Kolom yang tersedia: {inputs.keys()}")

    if inputs[FEATURE_KEY] is None:
        raise ValueError("ERROR: Nilai FEATURE_KEY adalah None!")

    if inputs[LABEL_KEY] is None:
        raise ValueError("ERROR: Nilai LABEL_KEY adalah None!")

    outputs = {}
    LABEL_NAME = LABEL_KEY + "_xf"
    FEATURE_NAME = FEATURE_KEY + "_xf"

    outputs[FEATURE_NAME] = tf.strings.lower(inputs[FEATURE_KEY])
    outputs[LABEL_NAME] = tf.cast(inputs[LABEL_KEY], tf.int64)

    return outputs
