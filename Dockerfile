FROM tensorflow/serving

# Copy model dan konfigurasi
COPY ./ouputs/serving_model /models/hatespeech-prediction-model
COPY ./config /model_config

# Set environment variables
ENV MODEL_NAME=hatespeech-prediction-model
# ENV MODEL_BASE_PATH=/models
ENV MONITORING_CONFIG="/model_config/prometheus.config"

RUN echo '#!/bin/bash \n\n\
tensorflow_model_server \
--rest_api_port=$PORT \
--model_name=${MODEL_NAME} \
--model_base_path=${MODEL_BASE_PATH}/${MODEL_NAME} \
"$@"' > /usr/bin/tf_serving_entrypoint.sh \
&& chmod +x /usr/bin/tf_serving_entrypoint.sh
