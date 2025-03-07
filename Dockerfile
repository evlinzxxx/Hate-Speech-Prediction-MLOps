FROM tensorflow/serving:latest

# Copy model dan konfigurasi
COPY ./serving_model/hatespeech-prediction-model/ /models/hatespeech-prediction-model
COPY ./config/prometheus.config /model_config/prometheus.config

# Set environment variables
ENV MODEL_NAME=hatespeech-prediction-model
ENV MODEL_BASE_PATH=/models
ENV MONITORING_CONFIG=/model_config/prometheus.config
ENV PORT=8501

# Expose ports
EXPOSE 8500 8501

# Modify entrypoint script
RUN echo '#!/bin/bash \n\
tensorflow_model_server --port=8500 --rest_api_port=${PORT} --rest_api_address=0.0.0.0 \
--model_name=${MODEL_NAME} --model_base_path=${MODEL_BASE_PATH}/${MODEL_NAME} \
--monitoring_config_file=${MONITORING_CONFIG} \
"$@"' > /usr/bin/tf_serving_entrypoint.sh \
&& chmod +x /usr/bin/tf_serving_entrypoint.sh

# Set entrypoint
ENTRYPOINT ["/usr/bin/tf_serving_entrypoint.sh"]
