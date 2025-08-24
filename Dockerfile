FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000 8501

# Start supervisord to run both API and Streamlit frontend
CMD ["supervisord", "-c", "supervisord.conf"]