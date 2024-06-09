FROM python:3.12-slim

WORKDIR /app

# Copier le fichier requirements.txt et le dossier packages
COPY requirements.txt .

# Installer les dépendances nécessaires pour OpenCV
RUN apt-get update && apt-get install -y libgl1-mesa-glx libglib2.0-0

# Installer les dépendances Python
RUN pip install -r requirements.txt

# Copier le reste de l'application
COPY . .

# Définir la commande pour démarrer l'application
CMD ["python", "app.py"]