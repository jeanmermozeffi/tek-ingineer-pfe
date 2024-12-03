# Utiliser l'image Python comme base
FROM python:3.12.5

# Installer les dépendances nécessaires pour PostgreSQL et d'autres compilations
RUN apt-get update && apt-get install -y \
    libpq-dev \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Installer pip, setuptools et wheel pour éviter les problèmes de dépendances
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Copier le fichier des dépendances Python
COPY requirements.txt /tmp

# Installer les dépendances Python
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Commande pour démarrer le serveur MLflow (vous pouvez aussi l'exécuter dans docker-compose)
CMD mlflow server \
    --backend-store-uri postgresql://${POSTGRES_USER}:${POSTGRES_PASSWORD}@${MLFLOW_BACKEND_STORE}:${POSTGRES_PORT}/${POSTGRES_DB} \
    --default-artifact-root /${MLFLOW_ARTIFACT_STORE} \
    --host ${MLFLOW_TRACKING_SERVER_HOST} \
    --port ${MLFLOW_TRACKING_SERVER_PORT}
