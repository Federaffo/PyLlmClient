# Usa un'immagine base di Python
FROM python:3.9-slim

# Imposta la directory di lavoro
WORKDIR /app

# Copia i file di requirements (se esistono) e installa le dipendenze
COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copia il resto del codice dell'applicazione
COPY . .

# Comando di default per eseguire i test
CMD ["pytest", "tests"]
