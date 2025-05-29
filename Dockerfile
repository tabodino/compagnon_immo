FROM python:3.12-slim

WORKDIR /app

COPY src/ /app/src/
COPY models/ /app/models/
COPY requirements.txt /app/
COPY README.md /app/


RUN pip install --no-cache-dir -r requirements.txt


ENV PYTHONPATH="/app/src"


# ENTRYPOINT ["/bin/sh"]

EXPOSE 8501


CMD ["streamlit", "run", "src/streamlit/app.py", "--server.port=8501", "--server.address=0.0.0.0"]

# docker run -d -p 8501:8501 --name compagnon_immo \
# -v $(pwd)/src:/app/src \
# -v $(pwd)/models:/app/models \
# compagnon-immo
