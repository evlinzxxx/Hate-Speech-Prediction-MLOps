FROM tensorflow/serving:latest

# Copy model dan konfigurasi
COPY ./serving_model/hatespeech-prediction-model/ /models/hatespeech-prediction-model
COPY ./config/prometheus.config /model_config/prometheus.config

# Set environment variables
ENV MODEL_NAME=hatespeech-prediction-model
ENV MONITORING_CONFIG=/model_config/prometheus.config
ENV PORT=8501

RUN echo '#!/bin/bash
tensorflow_model_server \
  --rest_api_port="$PORT" \
  --rest_api_address="0.0.0.0" \
  --model_name=${MODEL_NAME} \
  --model_base_path=${MODEL_BASE_PATH}/${MODEL_NAME} \
  "$@"' > /usr/bin/tf_serving_entrypoint.sh \
&& chmod +x /usr/bin/tf_serving_entrypoint.sh
