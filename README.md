Compteur de Personnes
Avec Hailo AI sur Raspberry Pi 5
Documentation technique V60
Pipeline GStreamer + supervision.LineZone
Mars 2026
 
1. Introduction
Ce document décrit le système de détection et comptage de personnes base sur le Raspberry Pi 5 avec accélérateur Hailo AI (Hailo8, 26 TOPS). Le système utilise le pipeline GStreamer officiel de Hailo avec la bibliothèque supervision de Roboflow pour un comptage fiable par franchissement de ligne.
Le système supporte deux modes de fonctionnement : vue latérale (ligne verticale, comptage unidirectionnel ou bidirectionnel) et vue plongeante (ligne horizontale, comptage IN/OUT).
1.1 Historique du projet
Le projet a évolué à travers plusieurs itérations majeures :
Version	Résultat	Approche
V44-V45	2 passages	InferVStreams manuel + tiling + tracker maison
V49	9 passages	Seuil 0.3, centroide 120px, full-frame
V50-V52	8 passages	Filtres temporels, NMS agressive, déduplication
V60	Fiable	Pipeline GStreamer officiel + hailotracker + supervision
La V60 représente une réécriture complète, abandonnant l'approche manuelle (InferVStreams + tiling) au profit du pipeline GStreamer officiel qui résout tous les problèmes de multi-détection.
 
2. Architecture
2.1 Pipeline de traitement
Le flux de données suit le pipeline GStreamer suivant :
Camera/Video -> GStreamer pipeline:
  -> hailonet (inference sur Hailo8)
  -> hailofilter (post-process NMS natif)
  -> hailotracker (tracking integre)
  -> callback Python (extraction detections -> supervision.LineZone -> comptage)
  -> hailooverlay (dessin des bbox)
  -> affichage
2.2 Comparaison ancien vs nouveau pipeline
Aspect	V44-V52 (ancien)	V60 (actuel)
Inférence	InferVStreams manuel	GStreamer hailonet
Résolution	Tiling 320x320	Résolution native
Tracking	Maison (IoU + Kalman)	hailotracker (Hailo intègre)
NMS	Manuel (Python)	Hardware (Hailo NMS)
Comptage	Intersection de segments	supervision.LineZone
Multi-détection	Problème majeur	Résolu (1 track = 1 personne)
Code	~900 lignes	~480 lignes
2.3 Modèle de détection
Le pipeline GStreamerDetectionApp utilise automatiquement un modèle COCO (80 classes) adapte au hardware : yolov8m pour Hailo8 (26 TOPS) ou yolov8s pour Hailo8L (13 TOPS). Le filtrage par classe 'person' est effectué dans le callback Python.
 
3. Hardware requis
•	Raspberry Pi 5 (8 Go RAM recommande)
•	Hailo AI HAT+ (Hailo8 26 TOPS ou Hailo8L 13 TOPS)
•	Camera USB (640x480 YUYV/MJPG) ou Pi Camera Module
•	Alimentation USB-C 27W officielle
•	PCIe configure en Gen3 pour performances optimales
4. Installation
4.1 Mise à jour du système
sudo apt update && sudo apt full-upgrade -y
sudo rpi-eeprom-update -a
sudo reboot
4.2 Installation Hailo
sudo apt install hailo-all -y
sudo reboot
En cas de problème avec hailo-tappas-core (statut 'pi' au lieu de 'ii') :
sudo apt install --reinstall hailo-tappas-core
4.3 Vérification de l'installation
hailortcli fw-control identify
gst-inspect-1.0 hailotools
dpkg -l | grep hailo
La commande gst-inspect-1.0 doit lister hailotracker, hailonet, hailooverlay parmi les éléments disponibles.
4.4 Installation du projet
git clone https://github.com/hailo-ai/hailo-rpi5-examples.git
cd hailo-rpi5-examples
./install.sh
source setup_env.sh
pip install supervision paho-mqtt
4.5 Vérification du pipeline
python basic_pipelines/detection.py --input usb
Si cette commande affiche la vidéo avec des détections et des track_id, l'environnement est prêt.
 
5. Utilisation
5.1 Commande de base
python basic_pipelines/detectionPersonnesV60.py --input usb --use-frame
5.2 Options CLI complètes
Argument	Défaut	Description
--input	usb	Source : usb, rpi (Pi Camera), ou chemin fichier vidéo
--use-frame	off	Activer l'affichage vidéo avec overlay
--mode	latéral	latéral (ligne verticale) ou top (ligne horizontale)
--direction	ltr/ttb	ltr, rtl (latéral) ou ttb, btt (top)
--line-pos	0.5	Position de la ligne en ratio 0.0-1.0 (milieu du ROI)
--roi	aucun	Zone de comptage x1,y1,x2,y2 en pixels
--mirror	off	Retourner l'image horizontalement
--save-file	compteur_state.json	Fichier de sauvegarde JSON
--reset	off	Remettre les compteurs à zéro
--mqtt	aucun	Adresse du broker MQTT
--mqtt-port	1883	Port MQTT
--mqtt-topic	compteur/personnes	Topic MQTT de publication
--mqtt-interval	60	Intervalle de publication en secondes
--mqtt-user	aucun	Nom d'utilisateur MQTT
--mqtt-pass	aucun	Mot de passe MQTT
 
5.3 Exemples d'utilisation
Mode latéral (vue de cote)
# Gauche vers droite, sans ROI
python basic_pipelines/detectionPersonnesV60.py --input usb --use-frame

# Droite vers gauche avec ROI
python basic_pipelines/detectionPersonnesV60.py --input usb --use-frame \
  --direction rtl --roi 100,50,500,400
Mode top (vue plongeante)
# Haut vers bas
python basic_pipelines/detectionPersonnesV60.py --input usb --use-frame --mode top

# Bas vers haut avec ROI et ligne au tiers
python basic_pipelines/detectionPersonnesV60.py --input usb --use-frame \
  --mode top --direction btt --roi 50,50,600,600 --line-pos 0.3
Avec MQTT et persistance
python basic_pipelines/detectionPersonnesV60.py --input usb --use-frame \
  --mqtt 192.168.52.139 --mqtt-user monuser --mqtt-pass monpass \
  --save-file /home/pi/compteur_expo.json
Sur fichier vidéo
python basic_pipelines/detectionPersonnesV60.py \
  --input /home/pi/hailo-rpi5-examples/Marche2_P.mp4 --use-frame
 
6. Modes de comptage
6.1 Mode latéral
La caméra est placée sur le cote, les personnes passent devant de gauche à droite ou inversement. La ligne de comptage est verticale.
Direction	Description
ltr	Gauche vers droite : compte comme ENTREE
rtl	Droite vers gauche : compte comme ENTREE
6.2 Mode top
La caméra est placée au-dessus (vue plongeante), les personnes passent en dessous. La ligne de comptage est horizontale. Ce mode permet un comptage bidirectionnel IN/OUT.
Direction	Description
ttb	Haut vers bas : compte comme ENTREE
btt	Bas vers haut : compte comme ENTREE
6.3 Zone d'intérêt (ROI)
Le paramètre --roi x1,y1,x2,y2 définit un rectangle (affiche en magenta) dans lequel le comptage est actif. Seules les détections dont le centre est à l'intérieur du ROI sont prises en compte. La ligne de comptage (--line-pos) est calculée relativement au ROI, pas à l'image entière.
Exemple : --roi 100,50,500,400 --line-pos 0.5 place la ligne au milieu du ROI, soit à x=300 (et non x=320 milieu de l'image).
 
7. Affichage visuel
L'option --use-frame active l'affichage vidéo avec les éléments suivants :
•	Cadre vert : personne détectée, pas encore comptée
•	Cadre rouge : personne ayant franchi la ligne (comptée)
•	Ligne jaune : ligne de comptage (verticale ou horizontale selon le mode)
•	Fleche jaune : direction du comptage
•	Rectangle magenta : zone ROI (si définie)
•	Compteurs ENTREES / SORTIES en haut à gauche
•	ID de tracking unique par personne
L'option --mirror retourne l'image horizontalement (utile si la caméra est montée à l'envers). Les cadres, la ligne et la flèche s'adaptent automatiquement.
8. Persistance et reprise après crash
8.1 Sauvegarde automatique
Toutes les 20 secondes, si les compteurs IN ou OUT ont changé, le programme écrit un fichier JSON avec écriture atomique (.tmp puis os.replace) :
{
  "heure": "2026-03-14 15:30:45",
  "entrees": 12,
  "sorties": 3,
  "total": 15
}
8.2 Reprise après coupure
Au démarrage, le programme charge automatiquement le fichier de sauvegarde (s'il existe) et reprend le comptage à partir des valeurs sauvegardées. Les compteurs affiches sont toujours le total cumule (offset sauvegarde + session courante).
8.3 Remise à zéro
python basic_pipelines/detectionPersonnesV60.py --input usb --use-frame --reset
Cette commande supprime le fichier de sauvegarde et remet les compteurs a zéro.
 
9. Publication MQTT
9.1 Configuration
Le programme peut publier les compteurs vers un broker MQTT a intervalle régulier (60 secondes par défaut). La connexion supporte l'authentification par nom d'utilisateur et mot de passe.
python basic_pipelines/detectionPersonnesV60.py --input usb --use-frame \
  --mqtt 192.168.52.139 --mqtt-user pinode --mqtt-pass monpass
9.2 Format du message
Le message publie sur le topic (défaut : compteur/personnes) est au format JSON :
{
  "heure": "2026-03-14 16:45:30",
  "entrees": 12,
  "sorties": 3,
  "total": 15
}
9.3 Protocole et compatibilité
Le client utilise paho-mqtt avec le protocole MQTTv3.1.1. Le code est compatible avec paho-mqtt v1 et v2. La connexion est non bloquante (loop_start en thread arrière-plan) et tolérante aux erreurs : si le broker est injoignable, le comptage continue normalement.
9.4 Vérification de la réception
mosquitto_sub -h 192.168.52.139 -t "compteur/personnes" -v
 
10. Classes détectables (COCO)
Le modèle COCO utilise par GStreamerDetectionApp détecte 80 classes d'objets. Le callback filtre uniquement la classe 'person', mais il est possible de compter d'autres objets en modifiant le filtre.
10.1 Modifier le filtre de classe
Dans le callback app_callback, la ligne suivante filtre les personnes :
if label != "person":
    continue
Pour compter d'autres objets (ex: personnes + vélos) :
if label not in ("person", "bicycle"):
    continue
10.2 Liste des 80 classes COCO
ID	Label	ID	Label	ID	Label
0	person	27	tie	54	donut
1	bicycle	28	suitcase	55	cake
2	car	29	frisbee	56	chair
3	motorcycle	30	skis	57	couch
4	airplane	31	snowboard	58	potted plant
5	bus	32	sports ball	59	bed
6	train	33	kite	60	dining table
7	truck	34	baseball bat	61	toilet
8	boat	35	baseball glove	62	tv
9	traffic light	36	skateboard	63	laptop
10	fire hydrant	37	surfboard	64	mouse
11	stop sign	38	tennis racket	65	remote
12	parking meter	39	bottle	66	keyboard
13	bench	40	wine glass	67	cell phone
14	bird	41	cup	68	microwave
15	cat	42	fork	69	oven
16	dog	43	knife	70	toaster
17	horse	44	spoon	71	sink
18	sheep	45	bowl	72	refrigerator
19	cow	46	banana	73	book
20	elephant	47	apple	74	clock
21	bear	48	sandwich	75	vase
22	zebra	49	orange	76	scissors
23	giraffe	50	broccoli	77	teddy bear
24	backpack	51	carrot	78	hair drier
25	umbrella	52	hot dog	79	toothbrush
26	handbag	53	pizza		
 
11. Dépannage
11.1 Erreur 'not-negotiated' GStreamer
Cette erreur survient quand le pipeline demande un format/résolution que la caméra ne supporte pas. Vérifier les formats supportés :
v4l2-ctl --list-formats-ext -d /dev/video0
S'assurer aussi qu'aucun autre processus n'utilise la caméra :
sudo fuser -k /dev/video0
11.2 hailo-tappas-core non installe
Si le statut dpkg est 'pi' (partiellement installe) au lieu de 'ii' :
sudo apt install --reinstall hailo-tappas-core
dpkg -l | grep tappas  # verifier que le statut est 'ii'
11.3 Erreur MQTT 'Not authorized'
Si l'authentification échoue alors que MQTT Explorer fonctionne, forcer le protocole MQTTv3.1.1 dans le code :
client = mqtt.Client(..., protocol=mqtt.MQTTv311)
Verifier aussi qu'un autre client n'est pas connecté avec le même client_id.
11.4 Erreur d'import Python
Les imports Hailo varient selon la version du framework. Verifier avec :
head -15 ~/hailo-rpi5-examples/basic_pipelines/detection.py
Et adapter les imports de V60 en conséquence. La version actuelle utilise :
from hailo_apps.hailo_app_python.core.common.buffer_utils import ...
from hailo_apps.hailo_app_python.core.gstreamer.gstreamer_app import ...
from hailo_apps.hailo_app_python.apps.detection.detection_pipeline import ...
11.5 Log HailoRT
Le fichier hailort.log contient des informations précieuses sur la configuration du modèle, notamment le nombre de classes, le seuil NMS hardware (0.200), et le format de sortie. Consulter ce fichier en cas de problème de détection.
 
12. Structure du code V60
Le fichier detectionPersonnesV60.py (~480 lignes) est organisé en sections :
Variables globales
Configuration de la ligne, du mode, de la direction, du ROI, du miroir, de la sauvegarde et du MQTT. Initialisées dans __main__ et lues dans le callback.
Classe user_app_callback_class
Hérite de app_callback_class (Hailo). Contient les compteurs, les offsets de persistance, le client MQTT et les méthodes init_mqtt(), publish_mqtt_if_needed(), load_state(), save_state_if_needed().
Fonction app_callback
Callback GStreamer appelé pour chaque frame. Extrait les détections Hailo, filtre les personnes, applique le ROI, passe à supervision.LineZone pour le comptage, dessine l'overlay visuel, et déclenche la sauvegarde et la publication MQTT.
Bloc __main__
Parse les arguments CLI (custom_parser + parse_known_args pour cohabiter avec GStreamerDetectionApp), applique la configuration, charge la sauvegarde, initialise MQTT, et lance le pipeline.
