from pathlib import Path
import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib
import os
import argparse
import json
import time
import numpy as np
import cv2
import hailo
import supervision as sv

try:
    import paho.mqtt.client as mqtt
    MQTT_AVAILABLE = True
except ImportError:
    MQTT_AVAILABLE = False

from hailo_apps.hailo_app_python.core.common.buffer_utils import get_caps_from_pad, get_numpy_from_buffer
from hailo_apps.hailo_app_python.core.gstreamer.gstreamer_app import app_callback_class
from hailo_apps.hailo_app_python.apps.detection.detection_pipeline import GStreamerDetectionApp

# -----------------------------------------------------------------------------------------------
# Configuration globale (initialisée dans __main__, lue dans le callback)
# -----------------------------------------------------------------------------------------------
line_zone = None
MODE = "lateral"       # "lateral" = ligne verticale, "top" = ligne horizontale
LINE_POS = 0.5         # Position de la ligne en ratio (0.5 = milieu du ROI ou de l'image)
DIRECTION = "ltr"      # lateral: "ltr"/"rtl" — top: "ttb"/"btt"
ROI = None             # None ou (x1, y1, x2, y2) en pixels
MIRROR = False         # Retourner l'image horizontalement
SAVE_FILE = "compteur_state.json"
SAVE_INTERVAL = 20     # secondes
MQTT_BROKER = None     # None = desactive, ou "192.168.52.139"
MQTT_PORT = 1883
MQTT_TOPIC = "compteur/personnes"
MQTT_INTERVAL = 60     # secondes
MQTT_USER = None
MQTT_PASS = None

# -----------------------------------------------------------------------------------------------
# Classe callback utilisateur
# -----------------------------------------------------------------------------------------------
class user_app_callback_class(app_callback_class):
    def __init__(self):
        super().__init__()
        self.count_in = 0
        self.count_out = 0
        self.counted_ids = set()
        self.last_saved_in = 0
        self.last_saved_out = 0
        self.last_save_time = time.time()
        self.in_offset = 0   # offset charge depuis la sauvegarde
        self.out_offset = 0
        self.mqtt_client = None
        self.last_mqtt_time = 0

    def init_mqtt(self, broker, port, topic, username=None, password=None):
        """Initialiser la connexion MQTT"""
        if not MQTT_AVAILABLE:
            print("paho-mqtt non installe. Installer avec: pip install paho-mqtt")
            return
        try:
            try:
                self.mqtt_client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id="compteur_personnes")
            except (AttributeError, TypeError):
                self.mqtt_client = mqtt.Client(client_id="compteur_personnes")
            if username:
                self.mqtt_client.username_pw_set(username, password)
            self.mqtt_client.connect(broker, port, keepalive=60)
            self.mqtt_client.loop_start()
            self.mqtt_topic = topic
            auth_info = f" (auth: {username})" if username else ""
            print(f"MQTT connecte: {broker}:{port} topic={topic}{auth_info}")
        except Exception as e:
            print(f"Erreur connexion MQTT: {e}")
            self.mqtt_client = None

    def publish_mqtt_if_needed(self):
        """Publier les compteurs sur MQTT toutes les MQTT_INTERVAL secondes"""
        if self.mqtt_client is None:
            return
        now = time.time()
        if now - self.last_mqtt_time < MQTT_INTERVAL:
            return
        self.last_mqtt_time = now
        try:
            total_in = self.in_offset + self.count_in
            total_out = self.out_offset + self.count_out
            payload = json.dumps({
                "heure": time.strftime("%Y-%m-%d %H:%M:%S"),
                "entrees": total_in,
                "sorties": total_out,
                "total": total_in + total_out
            })
            self.mqtt_client.publish(self.mqtt_topic, payload, qos=1)
            # print("Publication MQTT +++++++++")
        except Exception as e:
            print(f"Erreur publication MQTT: {e}")

    def load_state(self, filepath):
        """Charger le comptage depuis le fichier JSON"""
        try:
            if os.path.exists(filepath):
                with open(filepath, 'r') as f:
                    data = json.load(f)
                self.in_offset = data.get("entrees", 0)
                self.out_offset = data.get("sorties", 0)
                self.last_saved_in = self.in_offset
                self.last_saved_out = self.out_offset
                print(f"Reprise du comptage: IN={self.in_offset} OUT={self.out_offset} "
                      f"(sauvegarde du {data.get('heure', '?')})")
                return True
        except Exception as e:
            print(f"Erreur lecture sauvegarde: {e} - redemarrage a zero")
        return False

    def save_state_if_needed(self, filepath):
        """Sauvegarder si les compteurs ont change et que 20s se sont ecoulees"""
        now = time.time()
        if now - self.last_save_time < SAVE_INTERVAL:
            return
        total_in = self.in_offset + self.count_in
        total_out = self.out_offset + self.count_out
        if total_in == self.last_saved_in and total_out == self.last_saved_out:
            return
        # Sauvegarde
        try:
            data = {
                "heure": time.strftime("%Y-%m-%d %H:%M:%S"),
                "entrees": total_in,
                "sorties": total_out,
                "total": total_in + total_out
            }
            tmp = filepath + ".tmp"
            with open(tmp, 'w') as f:
                json.dump(data, f, indent=2)
            os.replace(tmp, filepath)
            self.last_saved_in = total_in
            self.last_saved_out = total_out
            self.last_save_time = now
        except Exception as e:
            print(f"Erreur sauvegarde: {e}")

# -----------------------------------------------------------------------------------------------
# Callback appele par le pipeline GStreamer pour chaque frame
# -----------------------------------------------------------------------------------------------
def app_callback(pad, info, user_data):
    buffer = info.get_buffer()
    if buffer is None:
        return Gst.PadProbeReturn.OK

    user_data.increment()
    frame_count = user_data.get_count()

    format, width, height = get_caps_from_pad(pad)

    # Initialiser la ligne de comptage au premier frame
    global line_zone
    if line_zone is None and width is not None and height is not None:
        if MODE == "lateral":
            # Ligne verticale
            if ROI is not None:
                line_x = int(ROI[0] + (ROI[2] - ROI[0]) * LINE_POS)
                line_a = sv.Point(line_x, ROI[1])
                line_b = sv.Point(line_x, ROI[3])
            else:
                line_x = int(width * LINE_POS)
                line_a = sv.Point(line_x, 0)
                line_b = sv.Point(line_x, height)
            # ltr: start=haut, end=bas → IN = gauche vers droite
            # rtl: start=bas, end=haut → IN = droite vers gauche
            if DIRECTION == "ltr":
                line_start, line_end = line_a, line_b
            else:
                line_start, line_end = line_b, line_a
        else:
            # Mode top : ligne horizontale
            if ROI is not None:
                line_y = int(ROI[1] + (ROI[3] - ROI[1]) * LINE_POS)
                line_a = sv.Point(ROI[0], line_y)
                line_b = sv.Point(ROI[2], line_y)
            else:
                line_y = int(height * LINE_POS)
                line_a = sv.Point(0, line_y)
                line_b = sv.Point(width, line_y)
            # ttb: start=gauche, end=droite → IN = haut vers bas
            # btt: start=droite, end=gauche → IN = bas vers haut
            if DIRECTION == "ttb":
                line_start, line_end = line_a, line_b
            else:
                line_start, line_end = line_b, line_a

        line_zone = sv.LineZone(
            start=line_start,
            end=line_end,
            triggering_anchors=(sv.Position.CENTER, sv.Position.BOTTOM_CENTER)
        )
        if MODE == "lateral":
            dir_label = "gauche vers droite" if DIRECTION == "ltr" else "droite vers gauche"
            print(f"Mode lateral - Ligne verticale: x={line_a.x}, direction={dir_label}")
        else:
            dir_label = "haut vers bas" if DIRECTION == "ttb" else "bas vers haut"
            print(f"Mode top - Ligne horizontale: y={line_a.y}, direction={dir_label}")
        if ROI:
            print(f"ROI: ({ROI[0]},{ROI[1]}) vers ({ROI[2]},{ROI[3]})")

    # Recuperer le frame si --use-frame
    frame = None
    if user_data.use_frame and format is not None and width is not None and height is not None:
        frame = get_numpy_from_buffer(buffer, format, width, height)

    # Recuperer les detections Hailo
    roi_hailo = hailo.get_roi_from_buffer(buffer)
    hailo_detections = roi_hailo.get_objects_typed(hailo.HAILO_DETECTION)

    # Construire les arrays (personnes uniquement, filtrees par ROI)
    boxes_list = []
    conf_list = []
    class_list = []
    tracker_list = []

    for detection in hailo_detections:
        label = detection.get_label()
        if label != "person":
            continue

        confidence = detection.get_confidence()
        bbox = detection.get_bbox()

        track_id = 0
        track = detection.get_objects_typed(hailo.HAILO_UNIQUE_ID)
        if len(track) == 1:
            track_id = track[0].get_id()

        # Coordonnees pixels
        x1 = bbox.xmin() * width
        y1 = bbox.ymin() * height
        x2 = bbox.xmax() * width
        y2 = bbox.ymax() * height

        # Filtrage ROI : le centre de la bbox doit etre dans le rectangle
        if ROI is not None:
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
            if not (ROI[0] <= cx <= ROI[2] and ROI[1] <= cy <= ROI[3]):
                continue

        boxes_list.append([x1, y1, x2, y2])
        conf_list.append(confidence)
        class_list.append(0)
        tracker_list.append(track_id)

    # Comptage par franchissement de ligne
    if len(boxes_list) > 0 and line_zone is not None:
        detections = sv.Detections(
            xyxy=np.array(boxes_list),
            confidence=np.array(conf_list),
            class_id=np.array(class_list, dtype=int),
            tracker_id=np.array(tracker_list, dtype=int)
        )
        try:
            crossed_in, crossed_out = line_zone.trigger(detections)
            for i in range(len(tracker_list)):
                if crossed_in[i] or crossed_out[i]:
                    user_data.counted_ids.add(tracker_list[i])
        except (TypeError, ValueError):
            old_total = line_zone.in_count + line_zone.out_count
            line_zone.trigger(detections)
            if line_zone.in_count + line_zone.out_count > old_total:
                for tid in tracker_list:
                    user_data.counted_ids.add(tid)

        user_data.count_in = line_zone.in_count
        user_data.count_out = line_zone.out_count

    # Affichage
    if frame is not None:
        # Miroir horizontal si active
        if MIRROR:
            frame = cv2.flip(frame, 1)
            # Fonction pour inverser les coordonnees X
            def fx(x):
                return int(width - 1 - x)
        else:
            def fx(x):
                return int(x)

        # ROI (rectangle magenta)
        if ROI is not None:
            rx1, rx2 = fx(ROI[0]), fx(ROI[2])
            if rx1 > rx2:
                rx1, rx2 = rx2, rx1
            cv2.rectangle(frame, (rx1, ROI[1]), (rx2, ROI[3]), (255, 0, 255), 2)
            cv2.putText(frame, "ROI", (rx1 + 5, ROI[1] + 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)

        # Bbox colorees
        for i in range(len(boxes_list)):
            bx1_raw, by1, bx2_raw, by2 = [int(v) for v in boxes_list[i]]
            bx1, bx2 = fx(bx1_raw), fx(bx2_raw)
            if bx1 > bx2:
                bx1, bx2 = bx2, bx1
            tid = tracker_list[i]
            if tid in user_data.counted_ids:
                color = (0, 0, 255)  # Rouge = compte
            else:
                color = (0, 255, 0)  # Vert = pas encore compte
            cv2.rectangle(frame, (bx1, by1), (bx2, by2), color, 2)
            cv2.putText(frame, f"ID:{tid}", (bx1, by1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Ligne de comptage + fleche de direction
        if line_zone is not None:
            if MODE == "lateral":
                # Ligne verticale
                if ROI is not None:
                    raw_lx = int(ROI[0] + (ROI[2] - ROI[0]) * LINE_POS)
                    ly_top, ly_bottom = ROI[1], ROI[3]
                else:
                    raw_lx = int(width * LINE_POS)
                    ly_top, ly_bottom = 0, height
                lx = fx(raw_lx)
                cv2.line(frame, (lx, ly_top), (lx, ly_bottom), (0, 255, 255), 3)
                arrow_y = ly_top + 30
                if (DIRECTION == "ltr") != MIRROR:
                    cv2.arrowedLine(frame, (lx - 40, arrow_y), (lx + 40, arrow_y), (0, 255, 255), 2, tipLength=0.4)
                else:
                    cv2.arrowedLine(frame, (lx + 40, arrow_y), (lx - 40, arrow_y), (0, 255, 255), 2, tipLength=0.4)
            else:
                # Mode top : ligne horizontale
                if ROI is not None:
                    raw_ly = int(ROI[1] + (ROI[3] - ROI[1]) * LINE_POS)
                    lx_left = fx(ROI[0])
                    lx_right = fx(ROI[2])
                else:
                    raw_ly = int(height * LINE_POS)
                    lx_left = fx(0)
                    lx_right = fx(width)
                if lx_left > lx_right:
                    lx_left, lx_right = lx_right, lx_left
                cv2.line(frame, (lx_left, raw_ly), (lx_right, raw_ly), (0, 255, 255), 3)
                arrow_x = lx_left + 30
                if DIRECTION == "ttb":
                    cv2.arrowedLine(frame, (arrow_x, raw_ly - 30), (arrow_x, raw_ly + 30), (0, 255, 255), 2, tipLength=0.4)
                else:
                    cv2.arrowedLine(frame, (arrow_x, raw_ly + 30), (arrow_x, raw_ly - 30), (0, 255, 255), 2, tipLength=0.4)

        # Compteurs (offset sauvegarde + comptage courant)
        in_total = user_data.in_offset + (line_zone.in_count if line_zone else 0)
        out_total = user_data.out_offset + (line_zone.out_count if line_zone else 0)
        cv2.putText(frame, f"ENTREES: {in_total}", (10, 60),
                    cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(frame, f"SORTIES: {out_total}", (10, 95),
                    cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 0, 255), 2)
        cv2.putText(frame, f"Personnes: {len(boxes_list)}",
                    (10, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)

        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        user_data.set_frame(frame)

    # Sauvegarde periodique (toutes les 20s, uniquement si changement)
    user_data.save_state_if_needed(SAVE_FILE)

    # Publication MQTT periodique (toutes les 60s)
    user_data.publish_mqtt_if_needed()

    # Log periodique
    if frame_count % 60 == 0 and line_zone is not None:
        in_total = user_data.in_offset + line_zone.in_count
        out_total = user_data.out_offset + line_zone.out_count
        print(f"Frame {frame_count}: IN={in_total} OUT={out_total} "
              f"Personnes={len(boxes_list)}")

    return Gst.PadProbeReturn.OK

if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent.parent
    env_file = project_root / ".env"
    os.environ["HAILO_ENV_FILE"] = str(env_file)

    # Parser les arguments personnalises AVANT de lancer le pipeline
    import sys
    custom_parser = argparse.ArgumentParser(add_help=False)
    custom_parser.add_argument('--roi', type=str, default=None,
        help='Zone de comptage x1,y1,x2,y2 en pixels. Ex: --roi 100,50,500,400')
    custom_parser.add_argument('--mode', '-m', choices=['lateral', 'top'], default='lateral',
        help='Mode: lateral (ligne verticale) ou top (vue plongeante, ligne horizontale). Defaut: lateral')
    custom_parser.add_argument('--direction', '-d', choices=['ltr', 'rtl', 'ttb', 'btt'], default=None,
        help='Direction comptee comme ENTREE: ltr/rtl (lateral), ttb/btt (top). Defaut: ltr ou ttb selon mode')
    custom_parser.add_argument('--line-pos', type=float, default=0.5,
        help='Position de la ligne en ratio 0.0-1.0 dans le ROI ou l\'image (defaut: 0.5 = milieu)')
    custom_parser.add_argument('--mirror', action='store_true',
        help='Retourner l\'image horizontalement (effet miroir)')
    custom_parser.add_argument('--save-file', type=str, default='compteur_state.json',
        help='Fichier de sauvegarde JSON (defaut: compteur_state.json)')
    custom_parser.add_argument('--reset', action='store_true',
        help='Remettre les compteurs a zero (ignore la sauvegarde)')
    custom_parser.add_argument('--mqtt', type=str, default=None,
        help='Adresse du broker MQTT. Ex: --mqtt 192.168.52.139')
    custom_parser.add_argument('--mqtt-port', type=int, default=1883,
        help='Port MQTT (defaut: 1883)')
    custom_parser.add_argument('--mqtt-topic', type=str, default='compteur/personnes',
        help='Topic MQTT (defaut: compteur/personnes)')
    custom_parser.add_argument('--mqtt-interval', type=int, default=60,
        help='Intervalle de publication MQTT en secondes (defaut: 60)')
    custom_parser.add_argument('--mqtt-user', type=str, default=None,
        help='Nom d\'utilisateur MQTT')
    custom_parser.add_argument('--mqtt-pass', type=str, default=None,
        help='Mot de passe MQTT')
    custom_args, remaining = custom_parser.parse_known_args()
    sys.argv = [sys.argv[0]] + remaining

    # Appliquer la configuration
    MODE = custom_args.mode
    LINE_POS = custom_args.line_pos
    MIRROR = custom_args.mirror
    SAVE_FILE = custom_args.save_file

    # Direction par defaut selon le mode
    if custom_args.direction is None:
        DIRECTION = "ltr" if MODE == "lateral" else "ttb"
    else:
        DIRECTION = custom_args.direction
        # Validation coherence mode/direction
        if MODE == "lateral" and DIRECTION in ("ttb", "btt"):
            print(f"Attention: direction {DIRECTION} n'a de sens qu'en mode top. Utilisation de ltr.")
            DIRECTION = "ltr"
        elif MODE == "top" and DIRECTION in ("ltr", "rtl"):
            print(f"Attention: direction {DIRECTION} n'a de sens qu'en mode lateral. Utilisation de ttb.")
            DIRECTION = "ttb"
    if custom_args.mqtt:
        MQTT_BROKER = custom_args.mqtt
        MQTT_PORT = custom_args.mqtt_port
        MQTT_TOPIC = custom_args.mqtt_topic
        MQTT_INTERVAL = custom_args.mqtt_interval
        MQTT_USER = custom_args.mqtt_user
        MQTT_PASS = custom_args.mqtt_pass

    if custom_args.roi:
        try:
            parts = [int(x.strip()) for x in custom_args.roi.split(',')]
            if len(parts) != 4:
                raise ValueError("4 valeurs attendues")
            ROI = (parts[0], parts[1], parts[2], parts[3])
        except Exception as e:
            print(f"Format --roi invalide: {e}. Utiliser: --roi x1,y1,x2,y2")
            sys.exit(1)

    if MODE == "lateral":
        dir_label = "gauche vers droite" if DIRECTION == "ltr" else "droite vers gauche"
    else:
        dir_label = "haut vers bas" if DIRECTION == "ttb" else "bas vers haut"
    print(f"Mode: {MODE}")
    print(f"Direction: {dir_label}")
    print(f"Position ligne: {LINE_POS:.0%}")
    if MIRROR:
        print(f"Miroir: actif")
    if ROI:
        print(f"ROI: ({ROI[0]},{ROI[1]}) vers ({ROI[2]},{ROI[3]})")

    user_data = user_app_callback_class()

    # Charger la sauvegarde precedente (sauf si --reset)
    if custom_args.reset:
        print("Compteurs remis a zero")
        if os.path.exists(SAVE_FILE):
            os.remove(SAVE_FILE)
    else:
        user_data.load_state(SAVE_FILE)

    print(f"Sauvegarde: {SAVE_FILE} (toutes les {SAVE_INTERVAL}s)")

    # Connexion MQTT
    if MQTT_BROKER:
        user_data.init_mqtt(MQTT_BROKER, MQTT_PORT, MQTT_TOPIC, MQTT_USER, MQTT_PASS)

    app = GStreamerDetectionApp(app_callback, user_data)
    app.run()