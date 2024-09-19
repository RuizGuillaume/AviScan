import streamlit as st
import requests
from PIL import Image
import os
import streamlit.components.v1 as components

# Configuration de la page
st.set_page_config(page_title="Projet MLOps - Reconnaissance d'oiseaux", layout="wide")

# URLs des APIs
USER_API_URL = os.getenv("USER_API_URL", "http://user_api:5000")
ADMIN_API_URL = os.getenv("ADMIN_API_URL", "http://admin_api:5100")
API_KEY = "abcd1234"

if 'specie' not in st.session_state:
    st.session_state.specie = 0

if 'selected_species' not in st.session_state:
    st.session_state.selected_species = "Sélectionnez une espèce..."

if 'success' not in st.session_state:
    st.session_state.success = False

if 'api_accessible' not in st.session_state:
    st.session_state.api_accessible = False

# Empeche les options de fullscreen de s'afficher pour les images
hide_img_fs = '''
<style>
button[title="View fullscreen"]{
    visibility: hidden;}
</style>
'''
st.markdown(hide_img_fs, unsafe_allow_html=True)


# Fonction pour charger et redimensionner l'image avec une meilleure qualité
def load_and_resize_image(image_path, new_width):
    image = Image.open(image_path)
    width, height = image.size
    new_height = int(height * (new_width / width))
    return image.resize((new_width, new_height), Image.LANCZOS)


def get_api_status(token, username, password):
    """
    Renvoie l'état de l'api, inaccessible pendant le premier telechargement/preprocessing
    """
    if not st.session_state.api_accessible:
        try:
            headers = {"Authorization": f"Bearer {token}", "api-key": API_KEY}
            response = requests.get(
                f"{USER_API_URL}/get_status",
                headers=headers,
                data={"username": username, "password": password}
            )
            st.session_state.api_accessible = response.status_code == 200
        except Exception:
            st.session_state.api_accessible = False


# Sidebar pour la navigation
with st.sidebar:
    # Ajout du logo centré
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.image("logo.png", width=100)

    st.title("Navigation")
    page = st.selectbox(
        "Choisissez une page",
        [
            "Présentation du projet",
            "Technologies",
            "Schémas",
            "Résultats de l'entraînement",
            "Interface utilisateur (APIs)",
            "Conclusion"
        ]
    )

    st.markdown("---")

    mystyle = """
        <style>
            p {
                text-align: justify;
            }
        </style>
        """
    st.markdown(mystyle, unsafe_allow_html=True)
    st.info("Ce projet est développé dans le cadre d'une formation MLOps. Il démontre l'intégration de diverses \
            technologies pour créer un système de reconnaissance d'oiseaux robuste et scalable, \
            avec la participation active des utilisateurs.")

# Contenu principal
if page == "Présentation du projet":

    st.markdown("<h1 style='text-align: center;'>Projet MLOps - Reconnaissance d'oiseaux</h1>", unsafe_allow_html=True)

    # Chargement et affichage de l'image de couverture
    try:
        image = load_and_resize_image("oiseau_cover.jpg", 400)
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.image(image, caption="Reconnaissance d'oiseaux", use_column_width=True)
    except FileNotFoundError:
        st.warning("Image de couverture non trouvée. \
                   Veuillez vous assurer que 'oiseau_cover.jpg' est présent dans le répertoire du script.")

    # Création des onglets
    tabs = st.tabs(["Introduction", "Contexte", "Solution", "Architecture", "Participation des utilisateurs"])

    with tabs[0]:
        st.header("Introduction et présentation")
        st.write("""
        Bienvenue dans notre projet MLOps de reconnaissance d'oiseaux. Ce projet innovant vise à :
        - Identifier automatiquement les espèces d'oiseaux à partir d'images avec une haute précision
        - Utiliser des techniques avancées de deep learning, notamment EfficientNetB0
        - Appliquer les meilleures pratiques MLOps pour un déploiement robuste, scalable et maintenable
        - Impliquer activement les utilisateurs dans l'amélioration continue du modèle

        Notre solution combine l'intelligence artificielle de pointe, les principes MLOps, \
                 et la participation communautaire pour créer un outil puissant et évolutif.
        """)

    with tabs[1]:
        st.header("Contexte et problématique")
        st.write("""
        La biodiversité aviaire fait face à des défis sans précédent. Notre projet répond à ces enjeux en offrant :
        - Une identification rapide et précise des espèces d'oiseaux
        - Un outil participatif permettant aux utilisateurs de contribuer à l'enrichissement des données
        - Une plateforme d'apprentissage continu, s'adaptant aux nouvelles espèces et variations
        """)

    with tabs[2]:
        st.header("Solution")
        st.write("""
        Notre solution MLOps complète et participative comprend :
        1. API Utilisateur pour soumettre des images et recevoir des prédictions
        2. Système de contribution permettant aux utilisateurs d'enrichir le dataset
        3. Processus automatisé d'intégration des nouvelles données et de mise à jour du modèle
        4. Mécanisme de création de nouvelles classes pour les espèces non identifiées
        5. Plateforme communautaire pour l'identification collaborative des espèces inconnues
        """)

    with tabs[3]:
        st.header("Architecture")
        st.write("""
        Notre architecture basée sur Docker assure la portabilité de notre système et une importante scalabilité, \
                 tout en facilitant la contribution des utilisateurs :
        - Conteneurs spécialisés pour chaque composant du système
        - Intégration fluide des contributions des utilisateurs dans le pipeline de données
        - Mécanismes de validation et d'intégration des nouvelles espèces
        - Système de stockage et de traitement des images non identifiées
        """)

    with tabs[4]:
        st.header("Participation des utilisateurs")
        st.write("""
        Notre projet se distingue par son approche participative :
        1. Les utilisateurs peuvent soumettre leurs propres photos d'oiseaux
        2. Si l'espèce est reconnue, l'image enrichit le dataset existant
        3. Pour les nouvelles espèces, une nouvelle classe est créée automatiquement
        4. Les images d'espèces inconnues sont stockées pour une identification communautaire
        5. Ce processus permet une amélioration continue du modèle et une extension de sa couverture
        """)

if page == "Technologies":
    st.title("Technologies")
    st.write("")

    logo_list = [
        "python_logo.png",
        "tensorflow_logo.png",
        "mlflow_logo.png",
        "docker_logo.png",
        "github_logo.png"
    ]

    texts = [
        "Python fut utilisé pour rédiger l'intégralité des scripts nous permettant d'exécuter une pipeline CI/CD \
            complète, automatisant ainsi l'intégration continue et le déploiement de notre application avec une \
                grande efficacité.",
        "Notre modèle de classification a été développé avec TensorFlow, couplé à un modèle pré-entrainé \
            EfficientNetB0, qui assure la partie convolutive de notre réseau de neurones, \
                permettant une meilleure précision et une optimisation des performances lors de l'entraînement.",
        "Dans le cadre d'un suivi rigoureux des cycles d'entraînement de notre modèle, MLflow assure un historique \
            permanent des différentes versions de notre modèle, facilitant ainsi la gestion, la comparaison et \
                l'amélioration continue des résultats obtenus.",
        "Nous avons utilisé Docker pour isoler les différentes parties de notre application dans des conteneurs qui \
            fonctionnent ensemble de manière fluide, assurant ainsi non seulement la portabilité de notre application \
                sur diverses plateformes, mais aussi une gestion simplifiée de ses composants.",
        "GitHub nous offre une plateforme idéale pour travailler en groupe dans les meilleures conditions, \
            nous permettant également de vérifier le bon fonctionnement des différentes fonctions de notre projet, \
                tout en garantissant la collaboration entre les membres de l'équipe et le respect strict de la \
                    convention PEP 8."
    ]

    for i, text in enumerate(texts):
        col1, col2 = st.columns([1, 8])

        with col1:
            st.image(logo_list[i], width=65)

        with col2:
            st.write(f"###### {texts[i]}")

        st.markdown("---")

elif page == "Schémas":

    choix_schema = st.radio(
        "Choisissez un schéma :",
        [
            "Interaction utilisateur",
            "Architecture",
            'Pipeline'
        ]
    )

    if choix_schema == "Interaction utilisateur":
        st.markdown("<h1 style='text-align: center;'>Application</h1>", unsafe_allow_html=True)

        # On charge et affiche l'image SVG avec la possibilité de zoomer
        try:
            with open("application.svg", "r", encoding='utf-8') as svg_file:
                svg_content = svg_file.read()

        except FileNotFoundError:
            st.error("Le fichier 'application.svg' n'a pas été trouvé. \
                     Assurez-vous qu'il est présent dans le répertoire du script.")

        # On utilise un composant HTML personnalisé pour le zoom
        components.html(f"""
        <div id="svg-container" style="width: 100%; height: 600px; border: 1px solid #ddd; overflow: hidden;">
            {svg_content}
        </div>
        <script src="https://unpkg.com/panzoom@9.4.0/dist/panzoom.min.js"></script>
        <script>
            const element = document.getElementById('svg-container');
            const svgElement = element.querySelector('svg');
            svgElement.style.width = '100%';
            svgElement.style.height = '100%';
            panzoom(svgElement, {{
                maxZoom: 5,
                minZoom: 0.5,
                bounds: true,
                boundsPadding: 0.1
            }});
        </script>
        """, height=650)

        st.info("Utilisez la molette de la souris pour zoomer/dézoomer. \
                Cliquez et faites glisser pour vous déplacer dans l'image.")

        # Ajout d'explications détaillées sur l'application
        st.write("""
        ### Explication détaillée de l'application

        Notre architecture d'application intègre activement les contributions des utilisateurs :

        1. **Upload de l'image** :
        - L'image est d'abord envoyée par l'utilisateur sur le serveur, \
                 où elle est conservée dans un fichier temporaire.

        2. **Inférence** :
        - La prédiction de classe est effectuée en quelques centaines de millisecondes.
        - Le résultat comprends les 3 classes les plus probables ainsi que leurs scores.
        - Sont aussi affichées les images des classes en question pour aider l'utilisateur.

        3. **Feedback** :
        - Si l'utilisateur indique que la prédiction est correcte, on ajoute son image au dataset.
        - Si il indique que la prédiction est fausse mais connaît l'espèce, \
                 il l'indique et on ajoute son image au dataset.
        - Si il ne connaît pas l'image, on ajoute son image dans un dossier qui sera traité plus tard.

        Cette architecture supporte efficacement le flux de travail participatif, \
                 permettant une amélioration continue du modèle grâce aux contributions des utilisateurs.
        """)

    elif choix_schema == "Architecture":
        st.markdown("<h1 style='text-align: center;'>Architecture du projet</h1>", unsafe_allow_html=True)

        # On charge et affiche l'image SVG avec zoom
        try:
            with open("architecture.svg", "r", encoding='utf-8') as svg_file:
                svg_content = svg_file.read()

        except FileNotFoundError:
            st.error("Le fichier 'architecture.svg' n'a pas été trouvé. \
                     Assurez-vous qu'il est présent dans le répertoire du script.")

        # On corrige les symboles
        svg_content = svg_content.replace('DonnÃ©es', 'Données')
        svg_content = svg_content.replace('ModÃ¨les', 'Modèles')
        svg_content = svg_content.replace('traitÃ©es', 'traitées')
        svg_content = svg_content.replace('archivÃ©', 'archivé')

        # On utilise un composant HTML personnalisé pour le zoom
        components.html(f"""
        <div id="svg-container" style="width: 100%; height: 600px; border: 1px solid #ddd; overflow: hidden;">
            {svg_content}
        </div>
        <script src="https://unpkg.com/panzoom@9.4.0/dist/panzoom.min.js"></script>
        <script>
            const element = document.getElementById('svg-container');
            const svgElement = element.querySelector('svg');
            svgElement.style.width = '100%';
            svgElement.style.height = '100%';
            panzoom(svgElement, {{
                maxZoom: 5,
                minZoom: 0.5,
                bounds: true,
                boundsPadding: 0.1
            }});
        </script>
        """, height=650)

        st.info("Utilisez la molette de la souris pour zoomer/dézoomer. \
                Cliquez et faites glisser pour vous déplacer dans l'image.")

        # Ajout d'explications détaillées sur l'architecture
        st.write("""
        ### Explication détaillée de l'architecture Docker

        Notre architecture Docker intègre activement les contributions des utilisateurs :

        1. **Gestion des données** :
        - Entièrement autonome.
        - Responsable de l'acquisition, du nettoyage et de l'augmentation des données, \
                 y compris les nouvelles images soumises par les utilisateurs.
        - Gère la création de nouvelles classes pour les espèces non répertoriées.

        2. **Entraînement** :
        - Orchestre l'entraînement du modèle EfficientNetB0, s'adaptant aux nouvelles classes et données.
        - Utilise MLflow pour le suivi des expériences et la gestion des versions, \
                 crucial avec l'évolution constante du dataset.

        3. **Production** :
        - Héberge le modèle optimisé pour des prédictions en temps réel sur les nouvelles images soumises.
        - Peut être mis à jour régulièrement pour intégrer les améliorations basées sur les contributions des \
                 utilisateurs.

        4. **API client** :
        - Fournit des endpoints pour la soumission d'images, la récupération des prédictions, \
                 et la gestion des contributions utilisateurs.
        - Gère l'authentification et les autorisations pour sécuriser les contributions.

        5. **API administrative** :
        - Fournit des endpoints pour la gestion des données, des utilisateurs et des entraînements, \
                 pour la comparaison des métriques et le choix du modèle en production.

        6. **Interface** :
        - Interface Streamlit intuitive permettant aux utilisateurs de soumettre des images, voir les prédictions, \
                 et contribuer au dataset.
        - Offre des visualisations des performances du modèle et de l'évolution du dataset.

        7. **Monitoring** :
        - Surveille en temps réel les performances du modèle, \
                 particulièrement important avec l'ajout constant de nouvelles données.
        - Détecte les drifts potentiels causés par l'évolution du dataset.

        8. **MLflow** :
        - Centralise la gestion des expériences, des modèles et des métriques.
        - Crucial pour suivre l'évolution du modèle avec l'intégration continue de nouvelles données et classes.

        Cette architecture supporte efficacement le flux de travail participatif, \
                 permettant une amélioration continue du modèle grâce aux contributions des utilisateurs.
        """)

    elif choix_schema == "Pipeline":
        st.markdown("<h1 style='text-align: center;'>Pipeline CI/CD</h1>", unsafe_allow_html=True)

        # On charge et on affiche l'image SVG avec zoom
        try:
            with open("pipeline_ci_cd.svg", "r", encoding='utf-8') as svg_file:
                svg_content = svg_file.read()

        except FileNotFoundError:
            st.error("Le fichier 'pipeline_ci_cd.svg' n'a pas été trouvé. \
                     Assurez-vous qu'il est présent dans le répertoire du script.")

        # On utilise un composant HTML personnalisé pour le zoom
        components.html(f"""
        <div id="svg-container" style="width: 100%; height: 600px; border: 1px solid #ddd; overflow: hidden;">
            {svg_content}
        </div>
        <script src="https://unpkg.com/panzoom@9.4.0/dist/panzoom.min.js"></script>
        <script>
            const element = document.getElementById('svg-container');
            const svgElement = element.querySelector('svg');
            svgElement.style.width = '100%';
            svgElement.style.height = '100%';
            panzoom(svgElement, {{
                maxZoom: 5,
                minZoom: 0.5,
                bounds: true,
                boundsPadding: 0.1
            }});
        </script>
        """, height=650)

        st.info("Utilisez la molette de la souris pour zoomer/dézoomer. \
                Cliquez et faites glisser pour vous déplacer dans l'image.")

        st.write("""
        ### Explication détaillée de la pipeline CI/CD

        Notre pipeline permet une actualisation des données constantes ainsi \
                 qu'un modèle toujours performant et évolutif.

        1. **Arrivée de nouvelles données** :
        Les données sont téléchargées et corrigées dans un dossier dataset_raw (données brutes) puis déplacées \
                 dans un dossier dataset_clean une fois ayant passé l'étape du traitement des données (preprocessing).
        Tout nouvelle image arrive d'abord dans dataset_raw et se retrouve plus tard dans dataset_clean \
                 lors du preprocessing, qu'elle soit ajoutée par l'utilisateur ou qu'elle provienne de Kaggle.
        Un processus automatique chaque jour vérifie la disponibilité d'un nouveau dataset sur Kaggle. \
                 Si une version plus récente est disponible, elle est téléchargée.
        Dès qu'un nouveau dataset est téléchargé, le preprocessing des données se lance directement.

        2. **Traitement des données** :
        Lors du lancement d'un preproccessing, les labels du dataset Kaggle sont corrigés avec les noms corrects \
                 et qui correspondent à la liste qui permet d'ajouter une nouvelle espèce.
        Aussi, les dossiers train/test/valid sont répartis avec un équilibre de 70%/15%/15%.
        Le script s'assure de suivre les nouvelles images et classes :
        - Lorsqu'un certain nombre de nouvelles images (au moins l'équivalent de 1% de la totalité du dataset) \
                 est ajouté par les utilisateurs, un preprocessing est lancé.
        - Lorsqu'une nouvelle classe apparaît, elle ne passe jamais à travers le script de preprocessing, \
                 même si déclenché par les événements ci-dessus, tant qu'elle n'a pas suffisamment d'images.
        On considère qu'une classe est complète lorsqu'elle dispose au moins du même nombre d'images que la classe \
                 la plus petite de notre dataset. Lorsque la condition est remplie, \
                 un preprocessing se lance et cette classe se voit intégrée.
        Dès qu'un preprocessing est lancé, cela signifie que les données ont changé de manière suffisante pour \
                 déclencher une alerte aux administrateurs et les encourager à réentraîner le modèle.

        3. **Entraînement du modèle** :
        Un entraînement du modèle peut-être déclenché manuellement par un administrateur, par exemple car il a reçu \
                 une alerte indiquant une dérive du modèle ou l'arrivée de nombreuses nouvelles données.
        Lorsqu'un entraînement est terminé, l'administrateur est notifié et les informations relatives au nouveau \
                 modèle sont enregistrées dans MLflow.
        Aussi, une matrice de confusion couplée avec un rapport de classification est sauvegardée pour faire état de \
                 la performance du modèle à sa création.

        4. **MLflow** :
        Durant et après un entraînement, MLflow s'assure du suivi des métriques et de l'enregistrement des modèles \
                 ainsi que de la matrice évoquée plus haut.
        Il dispose d'une interface permettant d'afficher toutes les informations relatives aux entraînements \
                 et de faire les comparaisons nécessaires entre les modèles.
        Son intérêt est également de servir de moyen d'archive des modèles générés.

        5. **Évaluation du modèle** :
        Une fois un entraînement terminé, l'administrateur peut afficher une comparaison entre le modèle en \
                 production et celui qui vient d'être entraîné.
        Pour cela, il dispose des deux métriques les plus importantes, à savoir la validation_accuracy et la \
                 validation_loss, mais également d'une liste des noms et scores des 10 classes sur lesquelles le \
                 modèle est le moins performant, permettant de faciliter la prise de décision pour changer de modèle.

        6. **Déploiement** :
        Pour déployer le modèle, l'administrateur peut choisir parmi tous les modèles dans MLflow et simplement \
                 indiquer lequel passer en production.
        Le script chargé de l'inférence pour les utilisateurs charge en quelques secondes le nouveau modèle, \
                 sans interruption de service \
                 (seulement une attente si prédiction demandée en même temps que le changement de modèle).

        7. **Monitoring des performances** :
        Grâce à la matrice de confusion enregistrée avec chaque modèle, tous les jours, \
                 une nouvelle matrice est générée et comparée avec l'originale du modèle.
        Grâce à cela, il est possible de détecter un drift du modèle,\
                  soit une perte de performance sur certaines classes qu'il connaît à cause de nouvelles images qui \
                 peuvent être trop différentes de son entraînement.
        Un rapport est envoyé tous les jours avec les 10 meilleures et pires classes ainsi que les classes qui \
                 dérivent (positivement ou négativement, s'il y en a).

        8. **Gestion des conflits**
        Pour éviter des conflits entre les scripts de monitoring, de preprocessing et de training, \
                 chacun indique en permanence son état aux autres.
        Cela permet à l'un d'attendre que l'autre est fini pour se lancer \
                 et évite des erreurs ou corruption de données.
        """)

elif page == "Résultats de l'entraînement":
    st.title("Résultats et comparaison des modèles")

    # On affiche cette page seulement si l'utilisateur est connecté
    if 'admin_token' in st.session_state:
        try:
            headers = {
                "Authorization": f"Bearer {st.session_state.admin_token}",
                "api-key": API_KEY
            }

            # On récupère les résultats via l'API
            response = requests.get(f"{ADMIN_API_URL}/results", headers=headers, timeout=10)
            results = response.json()

            # On mets en page le JSON
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Dernier modèle entraîné :")
                st.markdown(
                    f"""<p style="font-size: 18px; display: inline;">Identifiant du run MLFlow : \
                    </p> <p style="font-size: 18px; color: #0da0e4; display: inline;"> \
                    {results['latest_run_id']}</p>""",
                    unsafe_allow_html=True
                )
                st.markdown(
                    f"""<p style="font-size: 18px; display: inline;">Précision (sur le set de validation) : \
                    </p> <p style="font-size: 18px; color: #0da0e4; display: inline;"> \
                    {results['latest_run_val_accuracy']}</p>""",
                    unsafe_allow_html=True
                )
                st.markdown(
                    f"""<p style="font-size: 18px; display: inline;">Perte (sur le set de validation) : \
                    </p> <p style="font-size: 18px; color: #0da0e4; display: inline;"> \
                    {results['latest_run_val_loss']}</p>""",
                    unsafe_allow_html=True
                )
                st.markdown(
                    """<b style="font-size: 18px; display: inline;"> </br> Les 10 pires classes : </b>""",
                    unsafe_allow_html=True
                )
                for classe, score in results['latest_run_worst_f1_scores'].items():
                    st.write(
                        f"""<p style="font-size: 18px; display: inline;">{classe}</p> : \
                        <p style="font-size: 18px; color: #0da0e4; display: inline;">{score} </p>""",
                        unsafe_allow_html=True
                    )
            with col2:
                st.subheader("Modèle actuellement utilisé :")
                st.markdown(
                    f"""<p style="font-size: 18px; display: inline;">Identifiant du run MLFlow : \
                    </p> <p style="font-size: 18px; color: #0da0e4; display: inline;"> \
                    {results['main_model_run_id']}</p>""",
                    unsafe_allow_html=True
                )
                st.markdown(
                    f"""<p style="font-size: 18px; display: inline;">Précision (sur le set de validation) : \
                    </p> <p style="font-size: 18px; color: #0da0e4; display: inline;"> \
                    {results['main_model_val_accuracy']}</p>""",
                    unsafe_allow_html=True
                )
                st.markdown(
                    f"""<p style="font-size: 18px; display: inline;">Perte (sur le set de validation) : \
                    </p> <p style="font-size: 18px; color: #0da0e4; display: inline;"> \
                    {results['main_model_val_loss']}</p>""",
                    unsafe_allow_html=True
                )
                st.markdown(
                    """<b style="font-size: 18px; display: inline;"> </br> Les 10 pires classes : </b>""",
                    unsafe_allow_html=True
                )
                for classe, score in results['main_model_worst_f1_scores'].items():
                    st.write(
                        f"""<p style="font-size: 18px; display: inline;">{classe}</p> : \
                        <p style="font-size: 18px; color: #0da0e4; display: inline;">{score} </p>""",
                        unsafe_allow_html=True
                    )

            # On ajoute un lien vers l'interface MLflow
            st.markdown(
                "</br>[Accéder à plus de détails sur l'interface MLFlow](http://localhost:5200)",
                unsafe_allow_html=True
            )

        except Exception as e:
            st.error(f"Impossible de communiquer avec l'API administrateur : {str(e)}")
            st.info("Vérifiez que tous les conteneurs sont en cours d'exécution \
                    et que les ports sont correctement configurés.")
    else:
        st.warning("Veuillez vous connecter en tant qu'administrateur pour accéder aux résultats MLflow.")
        st.info("Allez dans l'onglet 'Interface utilisateur (APIs)' et connectez-vous en tant qu'admin.")

elif page == "Interface utilisateur (APIs)":
    st.title("Interface utilisateur (APIs)")

    # Permet de choisir à quelle API se connecter
    api_choice = st.radio("Choisissez une API", ("Utilisateur", "Admin"))

    if api_choice == "Utilisateur":
        st.subheader("API Utilisateur")

        # Système d'authentification
        username = st.text_input("Nom d'utilisateur")
        password = st.text_input("Mot de passe", type="password")
        if st.button("Se connecter"):
            if 'user_token' not in st.session_state:
                try:
                    # On vérifie les indentifiants auprès de l'API
                    response = requests.post(f"{USER_API_URL}/token",
                                             data={"username": username, "password": password})
                    if response.status_code == 200:
                        st.success(f"Connecté en tant que {username}")
                        st.session_state.user_token = response.json()["access_token"]
                        get_api_status(st.session_state.user_token, username, password)
                    else:
                        st.error("Échec de l'authentification.")

                except Exception:
                    st.error("Impossible de se connecter à l'API utilisateur. \
                             Assurez-vous que les conteneurs Docker sont en cours d'exécution.")

        # Permet d'afficher un message de succès après la contribution de l'utilisateur
        if 'success' in st.session_state and st.session_state.success:
            st.toast("Merci pour votre précieuse contribution !")
            st.session_state.success = False

        # Système de prédiction si l'utilisateur ou l'administrateur est bien connecté
        if 'user_token' in st.session_state:
            # Si le premier preprocessing est terminé, l'api est accessible
            if st.session_state.api_accessible:
                uploaded_file = st.file_uploader("Choisissez une image d'oiseau", type=["jpg", "png"])

                if uploaded_file is not None:
                    # Vérifiez si l'image a déjà été traitée
                    if 'current_image' not in st.session_state or st.session_state.current_image != uploaded_file.name:
                        st.session_state.current_image = uploaded_file.name
                        st.session_state.prediction = None
                        st.session_state.class_images = None
                        st.session_state.feedback_given = False
                        st.session_state.feedback_step = None
                        st.session_state.selected_species = None

                    col1, col2, col3, col4 = st.columns([0.2, 0.2, 0.2, 0.22], vertical_alignment="bottom")

                    # On ouvre l'image pour l'afficher
                    image = Image.open(uploaded_file)
                    with col1:
                        st.image(image, caption="Image téléchargée")

                    # On envoie la prédiction au modèle
                    if st.session_state.prediction is None:
                        try:
                            files = {"file": ("image.jpg", uploaded_file.getvalue(), "image/jpeg")}
                            headers = {
                                "Authorization": f"Bearer {st.session_state.user_token}",
                                "api-key": API_KEY
                            }
                            # On fait une requête vers l'API
                            response = requests.post(f"{USER_API_URL}/predict", files=files, headers=headers)
                            st.session_state.prediction = response.json()
                        except Exception:
                            st.error(f"Impossible de communiquer avec l'API d'inférence : {response.json()}")

                    # On récupère les images des oiseaux
                    if st.session_state.class_images is None:
                        try:
                            st.session_state.class_images = []
                            headers = {"Authorization": f"Bearer {st.session_state.user_token}", "api-key": API_KEY}
                            for classe in st.session_state.prediction['predictions']:
                                response = requests.get(f"{USER_API_URL}/get_class_image",
                                                        params={'classe': classe},
                                                        headers=headers)
                                st.session_state.class_images.append(response.content)
                        except Exception:
                            st.error("Impossible de récupérer les images associées aux classes.")

                    # Affichage des prédictions
                    for i, col in enumerate([col2, col3, col4]):
                        with col:
                            try:
                                st.image(st.session_state.class_images[i])
                            except Exception:
                                st.error("Impossible d'afficher les images associées aux classes.")
                            st.markdown(
                                f"""<p style="font-size: 20px;"> \
                                {st.session_state.prediction['predictions'][i]} </p>""",
                                unsafe_allow_html=True
                            )
                            st.markdown(
                                f"""<p style="color: {'#079e20' if i==0 else '#d1ae29' if i==1 else '#b26a19'}; \
                                font-size: 20px;"> \
                                {str(round(st.session_state.prediction['scores'][i] * 100, 2)) + "%"} \
                                </p>""",
                                unsafe_allow_html=True
                            )

                    # Gestion des boutons de feedback
                    st.subheader("Une des prédictions est-elle correcte ?")
                    col1, col2, col3 = st.columns([0.1, 0.4, 0.5])

                    with col1:
                        if st.button("Oui", disabled=st.session_state.get('feedback_given', False)):
                            st.session_state.feedback_step = "correct_prediction"
                            st.session_state.selected_species = None
                    with col2:
                        if st.button(
                            "Non, mais je connais l'espèce correcte",
                            disabled=st.session_state.get('feedback_given', False)
                        ):
                            st.session_state.feedback_step = "known_species"
                            st.session_state.selected_species = None
                    with col3:
                        if st.button("Je ne suis pas sûr", disabled=st.session_state.get('feedback_given', False)):
                            st.session_state.feedback_step = "unsure"
                            data = {
                                "species": "NA",
                                "image_name": st.session_state.prediction['filename'],
                                "is_unknown": True
                            }
                            headers = {"Authorization": f"Bearer {st.session_state.user_token}", "api-key": API_KEY}
                            response = requests.post(f"{USER_API_URL}/add_image", headers=headers, data=data)
                            if response.status_code == 200:
                                st.session_state.success = True
                                st.session_state.feedback_given = True
                                st.rerun()
                            else:
                                st.error("Échec de l'envoi de l'image.")

                    # Gestion de la sélection d'espèce
                    if 'feedback_step' in st.session_state and not st.session_state.get('feedback_given', False):
                        if st.session_state.feedback_step == "correct_prediction":
                            st.session_state.selected_species = st.selectbox(
                                "Sélectionnez l'espèce correcte :",
                                ["Sélectionnez une espèce..."] + st.session_state.prediction['predictions']
                            )
                        elif st.session_state.feedback_step == "known_species":
                            if 'species_list' not in st.session_state:
                                headers = {"Authorization": f"Bearer {st.session_state.user_token}", "api-key": API_KEY}
                                response = requests.get(f"{USER_API_URL}/get_species", headers=headers)
                                st.session_state.species_list = response.json()['species']
                            st.session_state.selected_species = st.selectbox(
                                "Sélectionnez l'espèce correcte :",
                                ["Sélectionnez une espèce..."] + st.session_state.species_list
                            )

                        # Soumission de la sélection d'espèce
                        if (
                            st.session_state.selected_species
                            and st.session_state.selected_species != "Sélectionnez une espèce..."
                        ):
                            if st.button("Confirmer la sélection"):
                                data = {"species": st.session_state.selected_species,
                                        "image_name": st.session_state.prediction['filename'],
                                        "is_unknown": False
                                        }
                                headers = {"Authorization": f"Bearer {st.session_state.user_token}", "api-key": API_KEY}
                                response = requests.post(f"{USER_API_URL}/add_image", headers=headers, data=data)

                                if response.status_code == 200:
                                    st.session_state.success = True
                                    st.session_state.feedback_given = True
                                    st.rerun()
                                else:
                                    st.error("Échec de l'envoi de l'image.")

                # Réinitialiser l'état du feedback si aucun fichier n'est uploadé
                else:
                    st.session_state.feedback_given = False
                    st.session_state.feedback_step = None
                    st.session_state.selected_species = None

            # Si le premier preprocessing n'est pas terminé, l'api est inaccessible
            else:
                st.warning("Le premier démarrage nécessite le téléchargement d'un jeu de données. \
                    Attendez quelques minutes et rafraîchissez la page")
    else:
        st.subheader("API Admin")

        # Système d'authentification
        admin_username = st.text_input("Nom d'administrateur")
        admin_password = st.text_input("Mot de passe administrateur", type="password")

        if st.button("Se connecter (Admin)"):
            try:
                # On vérifie les indentifiants auprès de l'API
                response = requests.post(f"{ADMIN_API_URL}/token",
                                         data={"username": admin_username, "password": admin_password})
                if response.status_code == 200:
                    st.info(f"Connecté en tant qu'administrateur avec l'utilisateur : {admin_username}")
                    st.session_state.admin_token = response.json()["access_token"]
                    get_api_status(st.session_state.admin_token, admin_username, admin_password)
                else:
                    st.error(f"Impossible de se connecter à l'API administrateur, erreur {response.status_code}")

            except Exception:
                st.error("Impossible de se connecter à l'API administrateur. \
                         Assurez-vous que les conteneurs Docker sont en cours d'exécution.")

        # Si l'utilisateur est bien connecté, on affiche l'interface
        # Si le premier preprocessing est terminé, l'api est accessible
        if 'admin_token' in st.session_state:
            if st.session_state.api_accessible:
                # Permet d'ajouter un utilisateur
                st.subheader("Ajouter un utilisateur")
                new_username = st.text_input("Nouveau nom d'utilisateur")
                new_password = st.text_input("Nouveau mot de passe", type="password")
                is_admin = st.checkbox("Est administrateur")
                if st.button("Ajouter l'utilisateur"):
                    try:
                        headers = {
                            "Authorization": f"Bearer {st.session_state.admin_token}",
                            "api-key": API_KEY
                        }
                        data = {"new_username": new_username, "user_password": new_password, "is_admin": is_admin}
                        # On demande à l'API d'ajouter l'utilisateur
                        response = requests.post(f"{ADMIN_API_URL}/add_user", headers=headers, data=data)

                        st.info(response.json())

                    except Exception as e:
                        st.error(f"Impossible de communiquer avec l'API administrateur : {e}")

                st.write("---")

                # Lancer l'entraînement
                if st.button("Lancer l'entraînement"):
                    try:
                        headers = {
                            "Authorization": f"Bearer {st.session_state.admin_token}",
                            "api-key": API_KEY
                        }
                        # On demande à l'API de lancer l'entraînement
                        response = requests.get(f"{ADMIN_API_URL}/train", headers=headers)

                        st.info(response.json())
                    except Exception as e:
                        st.error(f"Impossible de communiquer avec l'API d'entraînement : {e}")

                st.write("---")

                st.write("Changer le modèle en production")
                run_id = st.text_input("Indiquez le run_id du modèle")
                if st.button("Valider"):
                    try:
                        headers = {
                            "Authorization": f"Bearer {st.session_state.admin_token}",
                            "api-key": API_KEY
                        }
                        data = {"run_id": run_id}
                        # On demande à l'API de changer le modèle en production
                        response = requests.post(f"{ADMIN_API_URL}/switchmodel", headers=headers, data=data)
                        st.info(response.json())
                    except Exception as e:
                        st.error(f"Impossible de communiquer avec l'API d'inférence : {e}")

            # Si le premier preprocessing n'est pas terminé, l'api est inaccessible
            else:
                st.warning("Le premier démarrage nécessite le téléchargement d'un jeu de données. \
                    Attendez quelques minutes et rafraîchissez la page")

elif page == "Conclusion":

    st.markdown("<h1 style='text-align: center;'>Projet MLOps - Reconnaissance d'oiseaux</h1>", unsafe_allow_html=True)

    # On charge et affiche l'image de couverture
    try:
        image = load_and_resize_image("oiseau_cover.jpg", 400)
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.image(image, caption="Reconnaissance d'oiseaux", use_column_width=True)
    except FileNotFoundError:
        st.warning("Image de couverture non trouvée. \
                   Veuillez vous assurer que 'oiseau_cover.jpg' est présent dans le répertoire du script.")

    st.write("""
            ### Conclusion

            Possibilités d'améliorations :

            - **Traitement des images inconnues**
            - **Volume sauvegardé dans le cloud**
            - **Affichage de meilleures images à l'inférence**
            - **Applications Android et IOS**

            Pour aller plus loin :

            - **Reconaissance du chant des oiseaux**


            """)

# Pied de page
st.sidebar.markdown("---")
st.sidebar.info("Développé par :\n- Maxence REMY-HAROCHE\n- Guillaume RUIZ\n- Yoni EDERY")
