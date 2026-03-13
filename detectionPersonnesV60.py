from pathlib import Path
import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib
import os
import argparse
import numpy as np
import cv2
import hailo
import supervision as sv

from hailo_apps.hailo_app_python.core.common.buffer_utils import get_caps_from_pad, get_numpy_from_buffer
from hailo_apps.hailo_app_python.core.gstreamer.gstreamer_app import app_callback_class
from hailo_apps.hailo_app_python.apps.detection.detection_pipeline import GStreamerDetectionApp

# -----------------------------------------------------------------------------------------------
# Configuration globale (initialisée dans __main__, lue dans le callback)
# -----------------------------------------------------------------------------------------------
line_zone = None
LINE_X_RATIO = 0.5
DIRECTION = "ltr"      # "ltr" = gauche vers droite, "rtl" = droite vers gauche
ROI = None             # None ou (x1, y1, x2, y2) en pixels

# -----------------------------------------------------------------------------------------------
# Classe callback utilisateur
# -----------------------------------------------------------------------------------------------
class user_app_callback_class(app_callback_class):
    def __init__(self):
        super().__init__()
        self.count_in = 0
        self.count_out = 0
        self.counted_ids = set()

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
        line_x = int(width * LINE_X_RATIO)
        # La direction de la ligne determine ce que supervision compte comme IN vs OUT
        # Ligne de haut en bas : IN = passage gauche vers droite
        # Ligne de bas en haut : IN = passage droite vers gauche
        if DIRECTION == "ltr":
            line_start = sv.Point(line_x, 0)
            line_end = sv.Point(line_x, height)
        else:
            line_start = sv.Point(line_x, height)
            line_end = sv.Point(line_x, 0)
        line_zone = sv.LineZone(
            start=line_start,
            end=line_end,
            triggering_anchors=(sv.Position.CENTER, sv.Position.BOTTOM_CENTER)
        )
        dir_label = "gauche vers droite" if DIRECTION == "ltr" else "droite vers gauche"
        print(f"Ligne: x={line_x}, hauteur={height}px, direction={dir_label}")
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
        # ROI (rectangle magenta)
        if ROI is not None:
            cv2.rectangle(frame, (ROI[0], ROI[1]), (ROI[2], ROI[3]), (255, 0, 255), 2)
            cv2.putText(frame, "ROI", (ROI[0] + 5, ROI[1] + 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)

        # Bbox colorees
        for i in range(len(boxes_list)):
            bx1, by1, bx2, by2 = [int(v) for v in boxes_list[i]]
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
            lx = int(width * LINE_X_RATIO)
            cv2.line(frame, (lx, 0), (lx, height), (0, 255, 255), 3)
            arrow_y = 30
            if DIRECTION == "ltr":
                cv2.arrowedLine(frame, (lx - 40, arrow_y), (lx + 40, arrow_y), (0, 255, 255), 2, tipLength=0.4)
            else:
                cv2.arrowedLine(frame, (lx + 40, arrow_y), (lx - 40, arrow_y), (0, 255, 255), 2, tipLength=0.4)

        # Compteurs
        in_count = line_zone.in_count if line_zone else 0
        out_count = line_zone.out_count if line_zone else 0
        cv2.putText(frame, f"ENTREES: {in_count}", (10, 60),
                    cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(frame, f"SORTIES: {out_count}", (10, 95),
                    cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 0, 255), 2)
        cv2.putText(frame, f"Personnes: {len(boxes_list)}",
                    (10, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)

        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        user_data.set_frame(frame)

    # Log periodique
    if frame_count % 60 == 0 and line_zone is not None:
        print(f"Frame {frame_count}: IN={line_zone.in_count} OUT={line_zone.out_count} "
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
    custom_parser.add_argument('--direction', '-d', choices=['ltr', 'rtl'], default='ltr',
        help='Direction comptee: ltr=gauche vers droite, rtl=droite vers gauche (defaut: ltr)')
    custom_parser.add_argument('--line-x', type=float, default=0.5,
        help='Position X de la ligne en ratio 0.0-1.0 (defaut: 0.5 = milieu)')
    custom_args, remaining = custom_parser.parse_known_args()
    sys.argv = [sys.argv[0]] + remaining

    # Appliquer la configuration
    DIRECTION = custom_args.direction
    LINE_X_RATIO = custom_args.line_x

    if custom_args.roi:
        try:
            parts = [int(x.strip()) for x in custom_args.roi.split(',')]
            if len(parts) != 4:
                raise ValueError("4 valeurs attendues")
            ROI = (parts[0], parts[1], parts[2], parts[3])
        except Exception as e:
            print(f"Format --roi invalide: {e}. Utiliser: --roi x1,y1,x2,y2")
            sys.exit(1)

    dir_label = "gauche vers droite" if DIRECTION == "ltr" else "droite vers gauche"
    print(f"Direction: {dir_label}")
    print(f"Position ligne: {LINE_X_RATIO:.0%}")
    if ROI:
        print(f"ROI: ({ROI[0]},{ROI[1]}) vers ({ROI[2]},{ROI[3]})")

    user_data = user_app_callback_class()
    app = GStreamerDetectionApp(app_callback, user_data)
    app.run()