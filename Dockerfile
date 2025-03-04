FROM tensorflow/serving:2.15.1

# Copy model dan konfigurasi
COPY serving_model/hatespeech-prediction-model/ /models/hatespeech-prediction-model
# COPY ./config/prometheus.config /model_config/prometheus.config

# Set environment variables
ENV MODEL_NAME=hatespeech-prediction-model
ENV MODEL_BASE_PATH=/models
ENTRYPOINT ["/usr/bin/tf_serving_entrypoint.sh"]
# ENV MONITORING_CONFIG=/model_config/prometheus.config
# ENV PORT=8501

# # Buat entrypoint script
# RUN echo '#!/bin/bash \n\n\
# env \n\
# tensorflow_model_server --port=8500 --rest_api_port=${PORT} \
# --model_name=${MODEL_NAME} --model_base_path=${MODEL_BASE_PATH} \
# --monitoring_config_file=${MONITORING_CONFIG} \
# "$@"' > /usr/bin/tf_serving_entrypoint.sh \
# && chmod +x /usr/bin/tf_serving_entrypoint.sh

# Gunakan entrypoint script sebagai default command
ENTRYPOINT ["/usr/bin/tf_serving_entrypoint.sh"]
