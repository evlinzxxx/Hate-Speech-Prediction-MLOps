
import os
from absl import logging
from typing import Text
from tfx.orchestration.beam.beam_dag_runner import BeamDagRunner
from tfx.orchestration import metadata, pipeline
from tfx.orchestration.metadata import sqlite_metadata_connection_config

PIPELINE_NAME = 'hatespeech-pipeline'

DATA_ROOT = 'data'
TRANSFORM_MODULE_FILE = 'modules/transform.py'
TRAINER_MODULE_FILE = 'modules/trainer.py'

OUTPUT_BASE = 'outputs'
serving_model_dir = os.path.join(OUTPUT_BASE, 'serving_model')
pipeline_root = os.path.join(OUTPUT_BASE, PIPELINE_NAME)
metadata_path = os.path.join(pipeline_root, 'metadata.sqlite')
metadata_config = sqlite_metadata_connection_config(metadata_path)


def initialize_local_pipeline(components, pipeline_root: Text) -> pipeline.Pipeline:
    """
    Initialize a local TFX pipeline.

    Args:
        components: A dictionary of TFX components to be included in the pipeline.
        pipeline_root: Root directory for pipeline output artifacts.

    Returns:
        A TFX pipeline.
    """
    logging.info(f'Pipeline root set to: {pipeline_root}')
    beam_args = [
    '--direct_running_mode=multi_processing',
    '--direct_num_workers=1',  # Kurangi jumlah worker untuk menghemat memori
    '--experiments=shuffle_mode=service',  # Mengurangi penggunaan memori Apache Beam
    '--runner=DirectRunner',
    '--max_num_records=10000'  # Batasi jumlah data yang diproses dalam satu waktu
    ]


    return pipeline.Pipeline(
        pipeline_name=PIPELINE_NAME,
        pipeline_root=pipeline_root,
        components=components,
        enable_cache=True,
        metadata_connection_config=metadata_config,
        beam_pipeline_args=beam_args
    )


if __name__ == '__main__':
    logging.set_verbosity(logging.INFO)

    from modules.components import init_components
    components = init_components(
        DATA_ROOT,
        training_module=TRAINER_MODULE_FILE,
        transform_module=TRANSFORM_MODULE_FILE,
        training_steps=5000,
        eval_steps=1000,
        serving_model_dir=serving_model_dir,
    )

    section = initialize_local_pipeline(components, pipeline_root)
    BeamDagRunner().run(pipeline = section)
