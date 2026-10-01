FROM python:3.12-slim
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /srv/frontend
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
RUN useradd -m appuser
USER appuser
EXPOSE 8501
CMD ["streamlit", "run", "streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
