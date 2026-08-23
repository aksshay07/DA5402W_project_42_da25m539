FROM python:3.11-slim

WORKDIR /app

RUN pip install --no-cache-dir \
    streamlit==1.37.0 \
    plotly==5.22.0 \
    pandas==2.2.2 \
    requests==2.32.3 \
    mlflow==2.13.0 \
    evidently==0.4.16 \
    pydantic==2.7.4 \
    "setuptools<81"
    

COPY dashboard/app.py ./dashboard/app.py
COPY scripts/simulate_drift.py ./scripts/simulate_drift.py
COPY monitoring/ ./monitoring/

EXPOSE 8501

CMD ["streamlit", "run", "dashboard/app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true"]
