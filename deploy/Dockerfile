# For a Docker Space (use if the Streamlit SDK is not offered for new Spaces).
FROM python:3.12-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
# Writable folder for the LLM cache; the Space's disk is reset on restart.
ENV CLAIMSSQL_CACHE_DIR=/tmp/claimssql_cache

EXPOSE 7860
CMD ["streamlit", "run", "app/streamlit_app.py", "--server.port=7860", "--server.address=0.0.0.0", "--server.headless=true"]
