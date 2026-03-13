Compteur de personne sur hailo et rpi5

Vérifier que l'exemple officiel fonctionne :
cd ~/hailo-rpi5-examples && source setup_env.sh
python basic_pipelines/detection.py --input usb

# 1. Mise à jour complète du système (firmware inclus)
sudo apt update && sudo apt full-upgrade -y
sudo rpi-eeprom-update -a

# 2. Reboot obligatoire après mise à jour firmware
sudo reboot

# 3. Installer le méta-paquet hailo-all (inclut hailort + tappas-core + dkms)
sudo apt install hailo-all -y

# 4. Reboot après installation
sudo reboot

# 5. Vérifier que tout est installé
hailortcli fw-control identify
gst-inspect-1.0 hailotools
dpkg -l | grep hailo

Le problème : hailo-tappas-core a le statut pi (partiellement installé / en attente) au lieu de ii (correctement installé). C'est pour ça que install.sh ne le détecte pas :
bashsudo apt install --reinstall hailo-tappas-core

Si ça échoue :
bashsudo dpkg --configure -a
sudo apt install -f
sudo apt install --reinstall hailo-tappas-core
Puis vérifier que le statut passe à ii :
bashdpkg -l | grep tappas
Une fois que c'est ii, relancer ./install.sh.

GStreamerDetectionApp gère TOUT le pipeline en interne (hailonet, hailotracker, hailooverlay, etc.)

Version V60. Résumé des changements (104 lignes ajoutées) :
Nouveaux arguments CLI :

Exemples d'utilisation :
bash# Basique (gauche→droite, pas de ROI, ligne au milieu)
python basic_pipelines/detectionPersonnesV60.py --input usb --use-frame

# Droite vers gauche
python basic_pipelines/detectionPersonnesV60.py --input usb --use-frame --direction rtl

# Avec ROI (seules les personnes dans le rectangle comptent)
python basic_pipelines/detectionPersonnesV60.py --input usb --use-frame --roi 100,50,500,400

# Tout combiné : ROI + direction + ligne à 40% de l'image
python basic_pipelines/detectionPersonnesV60.py --input usb --use-frame --roi 100,50,500,400 --direction rtl --line-x 0.4

--direction ltr|rtl : une flèche jaune sur la ligne indique la direction comptée. supervision.LineZone différencie IN et OUT selon l'orientation de la ligne (start→end), donc on inverse les points pour changer de direction.
--roi x1,y1,x2,y2 : rectangle magenta affiché, seules les détections dont le centre est à l'intérieur sont prises en compte pour le tracking et le comptage.