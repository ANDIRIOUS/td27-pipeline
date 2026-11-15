# Entorno fijado para regenerar la entrega. Los datos se montan, nunca se copian a la imagen.
FROM python:3.10.20-slim
COPY solution/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt
WORKDIR /r
CMD ["sh", "solution/run.sh"]
