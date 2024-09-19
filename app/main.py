from fastapi import (
    FastAPI,
    HTTPException,
    Depends,
    status,
    File,
    UploadFile,
    Header,
    Form,
)
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from typing import Optional
from pydantic import BaseModel
from datetime import datetime, timedelta
import uvicorn
import jwt
import os
import json
import pandas as pd
from dotenv import load_dotenv
import logging
from app.models.predictClass import predictClass
from fastapi.responses import FileResponse
from app.utils.github_uploader import upload_to_github

# Charger les variables d'environnement
load_dotenv()

# Configuration du logging
logging.basicConfig(
    filename="api.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

# Variables d'environnement
API_KEY = os.getenv("API_KEY")
SECRET_KEY = os.getenv("JWT_SECRET_KEY")
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30


# Charger les utilisateurs autorisés depuis le fichier JSON
def load_authorized_users():
    with open("authorized_users.json", "r") as f:
        return json.load(f)


AUTHORIZED_USERS = load_authorized_users()

app = FastAPI(
    title="Reconnaissance des oiseaux",
    description="API pour identifier l'espèce d'un oiseau à partir d'une photo.",
    version="0.1",
)

# on précharge Tensorflow et Cudnn (pour Nvidia) en important la classe et en faisant l'inférence d'une image
classifier = predictClass()
temp_image_path = "test_image.jpg"
classifier.predict(temp_image_path)


# Modèle Pydantic pour le token
class Token(BaseModel):
    access_token: str
    token_type: str


# Fonction pour créer un token JWT
def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


# Fonction pour vérifier le token JWT
def verify_token(token: str = Depends(OAuth2PasswordBearer(tokenUrl="/token"))):
    try:
        logging.info(f"Tentative de décodage du token: {token}")
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None or username not in AUTHORIZED_USERS:
            logging.warning("Le token ne contient pas de 'sub' valide")
            raise HTTPException(status_code=401, detail="Could not validate credentials")
        logging.info(f"Token validé pour l'utilisateur: {username}")
        return username
    except jwt.PyJWTError as e:
        logging.error(f"Erreur lors de la validation du token: {str(e)}")
        raise HTTPException(status_code=401, detail="Could not validate credentials")


# Fonction pour vérifier la clé API
def verify_api_key(api_key: str = Header(..., alias="api-key")):
    if api_key != API_KEY:
        logging.warning("Tentative d'accès avec une clé API invalide")
        raise HTTPException(status_code=403, detail="Invalid API Key")
    logging.info("Clé API validée")
    return api_key


# Fonction pour mettre à jour le fichier JSON des utilisateurs autorisés
def update_authorized_users(users):
    with open("authorized_users.json", "w") as f:
        json.dump(users, f, indent=4)


# Route pour obtenir un token
@app.post("/token", response_model=Token)
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends()):
    if form_data.username not in AUTHORIZED_USERS or form_data.password != ADMIN_PASSWORD:
        logging.warning(f"Tentative de connexion échouée pour l'utilisateur: {form_data.username}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(data={"sub": form_data.username}, expires_delta=access_token_expires)
    logging.info(f"Connexion réussie pour l'utilisateur: {form_data.username}")
    return {"access_token": access_token, "token_type": "bearer"}


# Route racine
@app.get("/")
async def root(api_key: str = Depends(verify_api_key), username: str = Depends(verify_token)):
    return {
        "message": "Bienvenue sur l'API de reconnaissance d'oiseaux",
        "user": username,
    }


# Route pour faire une prédiction
@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    api_key: str = Depends(verify_api_key),
    username: str = Depends(verify_token),
):
    logging.info(f"Prédiction demandée par l'utilisateur: {username}")
    try:
        # Créer le dossier tempImage s'il n'existe pas
        os.makedirs("tempImage", exist_ok=True)

        image_path = "tempImage/image.png"
        logging.info(f"Sauvegarde de l'image à: {image_path}")
        with open(image_path, "wb") as image_file:
            content = await file.read()
            image_file.write(content)

        logging.info("Début de la prédiction")
        meilleure_classe, highest_score = classifier.predict(image_path)
        logging.info(f"Prédiction terminée: {meilleure_classe}, score: {highest_score}")
        return {"prediction": meilleure_classe, "score": highest_score}
    except Exception as e:
        logging.error(f"Erreur lors de la prédiction: {str(e)}")
        logging.exception("Traceback complet:")
        raise HTTPException(status_code=500, detail=str(e))


# Route pour ajouter une image
@app.post("/add_image")
async def add_image(
    file: UploadFile = File(...),
    species: str = Form(...),
    is_new_species: bool = Form(False),
    is_unknown: bool = Form(False),
    api_key: str = Depends(verify_api_key),
    username: str = Depends(verify_token),
):
    try:
        content = await file.read()
        file_path = f"tempImage/{file.filename}"
        with open(file_path, "wb") as image_file:
            image_file.write(content)

        if is_unknown:
            github_url = upload_to_github(file_path)
            return {"status": "Image uploaded to GitHub", "url": github_url}
        elif is_new_species:
            new_class_path = f"data/train/{species}"
            os.makedirs(new_class_path, exist_ok=True)
            os.rename(file_path, f"{new_class_path}/{file.filename}")
            return {"status": f"New species '{species}' created and image added"}
        else:
            class_path = f"data/train/{species}"
            if not os.path.exists(class_path):
                raise HTTPException(status_code=400, detail=f"Species '{species}' does not exist")
            os.rename(file_path, f"{class_path}/{file.filename}")
            return {"status": f"Image added to existing species '{species}'"}
    except Exception as e:
        logging.error(f"Error adding image: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# Route pour obtenir la liste des espèces
@app.get("/get_species")
async def get_species(api_key: str = Depends(verify_api_key), username: str = Depends(verify_token)):
    df = pd.read_csv("./data/birds_list.csv")
    species_list = sorted(df["English"].tolist())
    return {"species": species_list}


# Route pour télécharger une image
@app.get("/get_class_image")
async def get_class_image(
    classe: str,
    api_key: str = Depends(verify_api_key),
    username: str = Depends(verify_token),
):
    dossier_classe = os.path.join("./data/test", classe)
    for name in os.listdir(dossier_classe):
        image_path = os.path.join(dossier_classe, name)
        return FileResponse(image_path, media_type="image/jpeg", filename=f"{classe}_image.jpg")
    raise HTTPException(status_code=404, detail="Image not found")


# Nouvelle route pour ajouter un utilisateur
@app.post("/add_user")
async def add_user(
    new_username: str = Header(...),
    api_key: str = Depends(verify_api_key),
    current_user: str = Depends(verify_token),
):
    global AUTHORIZED_USERS
    if new_username in AUTHORIZED_USERS:
        raise HTTPException(status_code=400, detail="User already exists")

    AUTHORIZED_USERS[new_username] = True
    update_authorized_users(AUTHORIZED_USERS)

    logging.info(f"Nouvel utilisateur ajouté par {current_user}: {new_username}")
    return {"status": "User added successfully"}


# Route pour obtenir la liste des utilisateurs autorisés
@app.get("/get_users")
async def get_users(api_key: str = Depends(verify_api_key), current_user: str = Depends(verify_token)):
    return {"authorized_users": list(AUTHORIZED_USERS.keys())}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
