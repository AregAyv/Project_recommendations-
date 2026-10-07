FROM python:3.11-slim

WORKDIR /app

COPY requirements-streamlit.txt .
RUN pip install --no-cache-dir -r requirements-streamlit.txt

COPY streamlit_app.py .

ENV PORT=8080
EXPOSE 8080

CMD ["sh", "-c", "streamlit run streamlit_app.py --server.port=${PORT} --server.address=0.0.0.0 --server.headless=true"]