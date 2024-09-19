from fastapi import (
    FastAPI,
    HTTPException,
    Depends,
    status,
    File,
    UploadFile,
    Header,
    Form
)
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from typing import Optional
from pydantic import BaseModel
from datetime import datetime, timedelta
import jwt
import os
import json
from dotenv import load_dotenv
import logging

# from app.models.predictClass import predictClass
from fastapi.responses import FileResponse, PlainTextResponse
import requests
import hashlib
import time
import pandas as pd

# Charger les variables d'environnement
load_dotenv()

# On lance le serveur FastAPI
app = FastAPI(
    title="Reconnaissance des oiseaux",
    description="API pour identifier l'espèce d'un oiseau à partir d'une photo.",
    version="0.1",
)

# On créer les différents chemins
volume_path = "volume_data"
dataset_clean_path = os.path.join(volume_path, "dataset_clean")
dataset_raw_path = os.path.join(volume_path, "dataset_raw")
log_folder = os.path.join(volume_path, "logs")
users_folder = os.path.join(volume_path, "authorized_users")
users_path = os.path.join(users_folder, "authorized_users.json")
temp_path = os.path.join(volume_path, "temp_images")
state_folder = os.path.join(volume_path, "containers_state")
preprocessing_state_path = os.path.join(state_folder, "preprocessing_state.txt")
temp_folder = os.path.join(volume_path, "temp_images")
unknown_images_path = os.path.join(volume_path, "unknown_images")


# On créer les dossiers si nécessaire
os.makedirs(log_folder, exist_ok=True)
os.makedirs(temp_path, exist_ok=True)

# On configure le logging pour les informations et les erreurs
logging.basicConfig(
    filename=os.path.join(log_folder, "user_api.log"),
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%d/%m/%Y %I:%M:%S %p",
)


# On définit les variables d'environnement
API_KEY = os.getenv("API_KEY")
SECRET_KEY = os.getenv("JWT_SECRET_KEY")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

# On attends que le container d'API ajoute les utilisateurs
while not os.path.exists(users_path):
    time.sleep(1)


# On charge les utilisateurs autorisés depuis le fichier JSON
def load_authorized_users():
    with open(users_path, "r") as f:
        return json.load(f)


# ----------------------------------------------------------------------------------------- #


# On utilise le modèle Pydantic pour le token
class Token(BaseModel):
    access_token: str
    token_type: str


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    """
    Permet de créer un token JWT.
    """
    # On copie les données dans une autre variable
    to_encode = data.copy()
    # Si une durée d'expiration est fournie, on l'ajoute
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        # Sinon, on définit une expiration par défaut à 15 minutes
        expire = datetime.utcnow() + timedelta(minutes=15)
    # On ajoute l'expiration aux données à encoder
    to_encode.update({"exp": expire})
    # On encode les données en indiquant la clé secrète et l'algorithme
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    logging.info("Token encodé avec succès !")
    # On renvoie le token encodé
    return encoded_jwt


def verify_token(token: str = Depends(OAuth2PasswordBearer(tokenUrl="/token"))):
    """
    Permet de vérifier le token JWT
    """
    try:
        # On tente de décoder le token
        logging.info(f"Tentative de décodage du token: {token}")
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        # On récupère le nom d'utilisateur dans le payload du token
        username: str = payload.get("sub")
        # On vérifie que l'utilisateur est bien autorisé
        user = load_authorized_users().get(username)
        # On renvoie une erreur si l'utilisateur n'est pas valide
        if user is None:
            logging.warning("Le token ne contient pas de 'sub' valide")
            raise HTTPException(
                status_code=401, detail="Impossible de valider les identifiants..."
            )
        # Sinon, on retourne le nom d'utilisateur et il n'y a pas d'erreur
        logging.info(f"Token validé pour l'utilisateur: {username}")
        return username
    except jwt.PyJWTError as e:
        logging.error(f"Erreur lors de la validation du token: {str(e)}")
        raise HTTPException(
            status_code=401, detail=f"Erreur lors de la validation du token: {str(e)}"
        )


def verify_api_key(api_key: str = Header(..., alias="api-key")):
    """
    Permet de vérifier la clé d'API
    """
    # On vérifie si la clé d'API est celle attendue
    # Si la clé n'est pas valide, on ne continue pas
    if api_key != API_KEY:
        logging.warning("Tentative d'accès avec une clé API invalide")
        raise HTTPException(
            status_code=403, detail="Tentative d'accès avec une clé API invalide"
        )
    # Sinon, on retourne la clé
    logging.info("Clé API validée !")
    return api_key


def update_authorized_users(users):
    """
    Permet de mettre à jour le fichier JSON des utilisateurs autorisés
    """
    # On ouvre le fichier contenant les utilisateurs
    with open(users_path, "w") as f:
        # On actualise le fichier
        json.dump(users, f, indent=4)


# ----------------------------------------------------------------------------------------- #


# Route pour obtenir un token
@app.post("/token", response_model=Token)
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends()):
    # On récupère l'utilisateur dans la liste des utilisateurs
    user = load_authorized_users().get(form_data.username)
    # Si l'utilisateur n'existe pas ou que son mot de passe est faux, on renvoie une erreur
    if user is None or form_data.password != user[1]:
        logging.warning(
            f"Tentative de connexion échouée pour l'utilisateur: {form_data.username}"
        )
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Nom d'utilisateur ou mot de passe incorrect",
            headers={"WWW-Authenticate": "Bearer"},
        )
    # On définit la durée d'expiration du token
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    # On crée le token d'accès avec les informations de l'utilisateur
    access_token = create_access_token(
        data={"sub": form_data.username}, expires_delta=access_token_expires
    )
    logging.info(f"Connexion réussie pour l'utilisateur: {form_data.username}")
    # On renvoie le token et son type
    return {"access_token": access_token, "token_type": "bearer"}


# Route pour tester le bon fonctionnement de l'API
# et retourner le nom d'utilisateur connecté
@app.get("/")
async def root(
    api_key: str = Depends(verify_api_key), username: str = Depends(verify_token)
):
    return {
        "message": "Bienvenue sur l'API de reconnaissance d'oiseaux",
        "user": username,
    }


# Route pour tester l'état du preprocessing et la disponibilité du reste de l'API
@app.get("/get_status")
async def get_status(
    api_key: str = Depends(verify_api_key), username: str = Depends(verify_token)
):
    try:
        # On vérifie que le dataset n'est pas en téléchargement
        with open(preprocessing_state_path, "r") as file:
            preprocessing_state = file.read()
        if preprocessing_state != "2":
            return PlainTextResponse(
                "L'API est prête, aucun téléchargement n'est en cours.",
                status_code=200
            )
        else:
            return PlainTextResponse(
                "L'API n'est pas prête, le premier téléchargement du dataset est en cours.",
                status_code=503
            )

    except Exception as e:
        logging.error(f"Erreur lors de la récupération de l'état du preprocessing: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Erreur lors de la récupération de l'état du preprocessing: {str(e)}",
        )


# Route pour faire une prédiction
@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    api_key: str = Depends(verify_api_key),
    current_user: str = Depends(verify_token),
):
    logging.info(f"Requête /predict reçue de l'utilisateur: {current_user}")
    try:
        # On lit le fichier envoyé
        content = await file.read()
        # On lui donne un nom unique basé sur son hash
        file_name = hashlib.sha256(content).hexdigest() + ".jpg"
        # On enregistre le fichier sur le volume
        file_path = os.path.join(temp_path, file_name)
        with open(file_path, "wb") as image_file:
            image_file.write(content)
        # On envoie la requête au conteneur d'inférence avec le nom de l'image qu'il doit récupérer
        response = requests.get(
            "http://inference:5500/predict", params={"file_name": file_name}
        )
        return response.json()

    except Exception as e:
        logging.error(f"Erreur lors de la prédiction: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Erreur lors de la prédiction: {str(e)}"
        )


# Route pour obtenir la liste des espèces
@app.get("/get_species")
async def get_species(
    api_key: str = Depends(verify_api_key), username: str = Depends(verify_token)
):
    try:
        # On vérifie que le dataset n'est pas en téléchargement
        # (état 2 du container de preprocessing)
        # et qu'il est donc présent pour y ajouter l'image
        with open(preprocessing_state_path, "r") as file:
            preprocessing_state = file.read()
        if preprocessing_state != "2":
            # On récupère la liste des espèces avec uniquement le nom anglais
            df = pd.read_csv(os.path.join(dataset_raw_path, "birds_list.csv"))
            species_list = sorted(df["English"].tolist())
            return {"species": species_list}
        else:
            return "La liste n'est pas encore présente, merci de patienter..."
    except Exception as e:
        logging.error(
            f"Erreur lors de la récupération de la liste des espèces: {str(e)}"
        )
        raise HTTPException(
            status_code=500,
            detail=f"Erreur lors de la récupération de la liste des espèces: {str(e)}",
        )


# Route pour télécharger l'image d'une espèce
@app.get("/get_class_image")
async def get_class_image(
    classe: str,
    api_key: str = Depends(verify_api_key),
    username: str = Depends(verify_token),
):
    try:
        # On vérifie que le dataset n'est pas en téléchargement
        # (état 2 du container de preprocessing)
        # et qu'il est donc présent pour y ajouter l'image
        with open(preprocessing_state_path, "r") as file:
            preprocessing_state = file.read()
        if preprocessing_state != "2":
            dossier_classe = os.path.join(dataset_raw_path, "train", classe)
            for name in os.listdir(dossier_classe):
                image_path = os.path.join(dossier_classe, name)
                return FileResponse(
                    image_path, media_type="image/jpeg", filename=f"{classe}_image.jpg"
                )
        else:
            return "Premier lancement -> le dataset est en cours de téléchargement/traitement, \
                vous serez notifié par mail de la fin du processus."
    except Exception as e:
        logging.error(
            f"Erreur lors de la récupération de l'image de l'espèce {classe}: {str(e)}"
        )
        raise HTTPException(
            status_code=500,
            detail=f"Erreur lors de la récupération de l'image de l'espèce {classe}: {str(e)}",
        )


# Route pour ajouter une image
@app.post("/add_image")
async def add_image(
    species: str = Form(...),
    image_name: str = Form(...),
    is_unknown: bool = Form(False),
    api_key: str = Depends(verify_api_key),
    current_user: str = Depends(verify_token),
):
    try:
        logging.info(f"Requête /add_image reçue de l'utilisateur: {current_user}")
        # On vérifie que le dataset n'est pas en téléchargement
        # (état 2 du container de preprocessing)
        # et qu'il est donc présent pour y ajouter l'image
        with open(preprocessing_state_path, "r") as file:
            preprocessing_state = file.read()
        if preprocessing_state != "2":
            # On créer le chemin vers l'image
            file_path = os.path.join(temp_folder, image_name)
            # Si la classe est inconnue, on l'ajoute dans le dossier des images inconnues
            if is_unknown:
                os.rename(file_path, f"{unknown_images_path}/{image_name}")
                return {"status": "Image ajoutée dans les images inconnues"}
            # Si la classe est connue, on l'ajoute dans train au bon endroit
            else:
                class_path = os.path.join(dataset_raw_path, f"train/{species}")
                if not os.path.exists(class_path):
                    os.makedirs(class_path, exist_ok=True)
                os.rename(file_path, f"{class_path}/{image_name}")
                return {"status": f"Image ajouteé dans l'espèce suivante: '{species}'"}
        else:
            return "Le dataset de base n'est pas encore présent, merci de patienter..."
    except Exception as e:
        logging.error(f"Une erreur est survenue lors de l'ajout de l'image: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Une erreur est survenue lors de l'ajout de l'image: {str(e)}",
        )
