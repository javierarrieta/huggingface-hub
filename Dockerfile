FROM python:3.13
RUN pip install huggingface_hub
RUN mkdir -p /app
ADD download_model.py /app
RUN chmod +x /app/download_model.py