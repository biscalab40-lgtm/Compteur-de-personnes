#!/usr/bin/env python3
"""
detectionPersonnesV48.py - V48 : DIAGNOSTIC BRUT des sorties Hailo
Compteur de personnes avec HailoRT 4.20.0 sur Raspberry Pi 5

But : comprendre exactement ce que le modèle retourne pour chaque tile
      et pourquoi les personnes au premier plan ne sont pas détectées.

Le diagnostic s'affiche pour les 3 premières frames uniquement.
Après la frame 3, le comportement est identique à V45.

Modes de fonctionnement :
  --mode top      : vue plongeante, ligne horizontale, comptage IN/OUT (défaut)
  --mode lateral  : vue de côté, ligne verticale, comptage unidirectionnel
  --skip N        : ne traiter qu'une frame sur N (performance)
"""

import cv2
import numpy as np
import json
import time
import os
import argparse
from pathlib import Path

# Import Hailo
try:
    from hailo_platform import (HEF, VDevice, HailoStreamInterface,
                               ConfigureParams, InputVStreamParams, OutputVStreamParams,
                               FormatType, InferVStreams, HailoSchedulingAlgorithm)
    HAILO_AVAILABLE = True
    print("✅ HailoRT chargé")

    import hailo_platform
    print(f"📦 Version: {hailo_platform.__version__}")
except ImportError as e:
    print(f"❌ HailoRT non disponible: {e}")
    exit(1)

class YOLOPostProcess:
    """Post-processing optimisé avec NumPy pour yolov8n.hef"""

    PERSON_CLASS_ID = 0  # ID de la classe 'person' dans COCO

    def __init__(self, elements_per_det=5):
        self.debug_done = False
        self.elements_per_det = elements_per_det

    def process_yolo_output(self, output, conf_threshold=0.5,
                          input_shape=(320,320), original_shape=(960,540),
                          scale=1.0, pad_x=0, pad_y=0):
        """Parse le buffer de sortie de manière vectorisée"""
        detections = []

        if output is None:
            return detections

        try:
            # 1. Conversion rapide en tableau 1D
            raw = np.array(output).flatten().astype(np.float32)

            if raw.size == 0:
                return detections

            # 2. Format fixe (5 = NMS sans class_id, 6 = NMS avec class_id)
            # Défaut: 5 floats [ymin, xmin, ymax, xmax, score] (Hailo NMS postprocess)
            elements_per_det = self.elements_per_det

            if raw.size % elements_per_det != 0:
                alt = 6 if elements_per_det == 5 else 5
                if raw.size % alt == 0:
                    elements_per_det = alt
                else:
                    print(f"⚠️ Buffer size {raw.size} incompatible avec format {elements_per_det} ou {alt}")
                    return detections

            if not self.debug_done:
                print(f"\n🔍 DEBUG - Taille du buffer: {raw.size} floats")
                print(f"  Format utilisé: {elements_per_det} valeurs par détection")

            # 3. Reshape matriciel (N détections, 5 ou 6 colonnes)
            boxes = raw.reshape(-1, elements_per_det)

            # 4. Filtrage vectorisé (Boolean Masking)
            scores = boxes[:, 4]
            valid_mask = scores >= conf_threshold

            # Filtrage de la classe si l'information est présente
            if elements_per_det == 6:
                classes = boxes[:, 5].astype(int)
                class_mask = classes == self.PERSON_CLASS_ID
                valid_mask = valid_mask & class_mask

            # Appliquer le masque pour ne garder que les bonnes détections
            valid_boxes = boxes[valid_mask]

            if len(valid_boxes) == 0:
                return detections

            # 5. Extraction vectorisée des coordonnées
            y_min_n = valid_boxes[:, 0]
            x_min_n = valid_boxes[:, 1]
            y_max_n = valid_boxes[:, 2]
            x_max_n = valid_boxes[:, 3]
            valid_scores = valid_boxes[:, 4]

            # 6. Conversions mathématiques vectorisées (Normalisé -> Pixels)
            x1 = ((x_min_n * input_shape[1]) - pad_x) / scale
            y1 = ((y_min_n * input_shape[0]) - pad_y) / scale
            x2 = ((x_max_n * input_shape[1]) - pad_x) / scale
            y2 = ((y_max_n * input_shape[0]) - pad_y) / scale

            # Cast en entiers
            x1, y1 = x1.astype(int), y1.astype(int)
            x2, y2 = x2.astype(int), y2.astype(int)

            # 7. Clamping vectorisé aux dimensions de l'image
            x1 = np.clip(x1, 0, original_shape[1] - 1)
            y1 = np.clip(y1, 0, original_shape[0] - 1)
            x2 = np.clip(x2, 0, original_shape[1] - 1)
            y2 = np.clip(y2, 0, original_shape[0] - 1)

            # 8. Filtrage des tailles de boîtes
            w = x2 - x1
            h = y2 - y1
            size_mask = (w > 10) & (h > 20)

            # 9. Construction de la liste finale
            # On ne boucle que sur les quelques détections valides restantes
            for i in range(len(x1)):
                if size_mask[i]:
                    detections.append({
                        'bbox': [int(x1[i]), int(y1[i]), int(x2[i]), int(y2[i])],
                        'conf': float(valid_scores[i]),
                        'center_y': int((y1[i] + y2[i]) // 2)
                    })

        except Exception as e:
            print(f"⚠️ Erreur post-processing NumPy: {e}")
            import traceback
            traceback.print_exc()

        if not self.debug_done:
            print(f"  📊 {len(detections)} personne(s) valide(s) détectée(s) après filtrage")
            self.debug_done = True

        return detections

class TemporalSmoother:
    """Lisse les détections sur plusieurs frames pour stabiliser les trajectoires"""

    def __init__(self, history_size=5, original_shape=(540, 960)):
        self.history = {}
        self.history_size = history_size
        self.original_shape = original_shape

    def update_shape(self, height, width):
        """Met à jour les dimensions de l'image"""
        self.original_shape = (height, width)

    def smooth(self, track_id, bbox, conf):
        """Ajoute une détection et retourne la version lissée"""
        if track_id not in self.history:
            self.history[track_id] = []

        self.history[track_id].append((bbox, conf))

        if len(self.history[track_id]) > self.history_size:
            self.history[track_id].pop(0)

        if len(self.history[track_id]) >= 3:
            recent = self.history[track_id][-3:]
            total_conf = sum(b[1] for b in recent)

            if total_conf > 0:
                avg_bbox = [
                    int(sum(b[0][0] * b[1] for b in recent) / total_conf),
                    int(sum(b[0][1] * b[1] for b in recent) / total_conf),
                    int(sum(b[0][2] * b[1] for b in recent) / total_conf),
                    int(sum(b[0][3] * b[1] for b in recent) / total_conf)
                ]
                avg_conf = total_conf / len(recent)
            else:
                avg_bbox = [
                    int(sum(b[0][0] for b in recent) / len(recent)),
                    int(sum(b[0][1] for b in recent) / len(recent)),
                    int(sum(b[0][2] for b in recent) / len(recent)),
                    int(sum(b[0][3] for b in recent) / len(recent))
                ]
                avg_conf = conf

            w = avg_bbox[2] - avg_bbox[0]
            h = avg_bbox[3] - avg_bbox[1]

            if (w < 10 or h < 20 or
                w > self.original_shape[1] or
                h > self.original_shape[0] or
                avg_bbox[0] < 0 or avg_bbox[1] < 0 or
                avg_bbox[2] > self.original_shape[1] or
                avg_bbox[3] > self.original_shape[0]):
                return bbox, conf

            return avg_bbox, avg_conf
        else:
            return bbox, conf

    def cleanup(self, active_tracks):
        """Nettoie l'historique des tracks disparus"""
        to_remove = [tid for tid in self.history if tid not in active_tracks]
        for tid in to_remove:
            del self.history[tid]

class TrajectoryTracker:
    """Tracker complet avec intersection de segments (In/Out) et Kalman Filter

    Modes :
      'top'     : ligne horizontale, comptage bidirectionnel (IN vers le bas, OUT vers le haut)
      'lateral' : ligne verticale, comptage unidirectionnel (file indienne)
    """

    def __init__(self, ligne_coords=None, mode='top', count_direction='left-to-right'):
        self.ligne = ligne_coords # Format: ((x1, y1), (x2, y2))
        self.mode = mode          # 'top' ou 'lateral'
        # Direction qui compte comme entrée en mode latéral :
        #   'left-to-right' : la personne entre en allant vers la droite
        #   'right-to-left' : la personne entre en allant vers la gauche
        self.count_direction = count_direction
        self.cnt_in = 0
        self.cnt_out = 0
        self.next_id = 0
        self.tracks = {}

        self.smoother = TemporalSmoother(history_size=5)
        self.iou_threshold = 0.3
        self.max_lost = 10
        self.kalman_enabled = True
        self.min_confidence = 0.3
        self.crossing_cooldown = 15  # frames minimum entre deux franchissements par track

    def _init_kalman(self, bbox):
        kf = cv2.KalmanFilter(6, 2)
        kf.measurementMatrix = np.array([[1, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0]], np.float32)
        kf.transitionMatrix = np.array([
            [1, 0, 0, 0, 1, 0], [0, 1, 0, 0, 0, 1], [0, 0, 1, 0, 0, 0],
            [0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 1]
        ], np.float32)
        kf.processNoiseCov = np.eye(6, dtype=np.float32) * 0.01
        cx, cy = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2
        w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
        kf.statePre = np.array([[cx], [cy], [w], [h], [0], [0]], np.float32)
        kf.statePost = np.array([[cx], [cy], [w], [h], [0], [0]], np.float32)
        return kf

    def _ccw(self, A, B, C):
        return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])

    def _intersect(self, A, B, C, D):
        """Algorithme d'intersection de segments (A-B traverse C-D)"""
        return self._ccw(A, C, D) != self._ccw(B, C, D) and \
               self._ccw(A, B, C) != self._ccw(A, B, D)

    def _compute_iou(self, box1, box2):
        x1, y1 = max(box1[0], box2[0]), max(box1[1], box2[1])
        x2, y2 = min(box1[2], box2[2]), min(box1[3], box2[3])
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - inter
        return inter / union if union > 0 else 0

    def _get_side(self, center):
        """Détermine de quel côté de la ligne se trouve le point"""
        if self.mode == 'lateral':
            # Ligne verticale : gauche / droite (on compare X)
            return 'right' if center[0] > self.ligne[0][0] else 'left'
        else:
            # Ligne horizontale : dessus / dessous (on compare Y)
            return 'below' if center[1] > self.ligne[0][1] else 'above'

    def _count_crossing(self, old_center, new_center, tid):
        """Comptabilise un franchissement selon le mode"""
        if self.mode == 'lateral':
            # Mode latéral : comptage unidirectionnel uniquement
            moving_right = new_center[0] > old_center[0]
            if self.count_direction == 'left-to-right' and moving_right:
                self.cnt_in += 1
                print(f"➡️ Passage! Total: {self.cnt_in}")
                self.tracks[tid]['counted'] = True
                self.tracks[tid]['frames_since_crossing'] = 0
            elif self.count_direction == 'right-to-left' and not moving_right:
                self.cnt_in += 1
                print(f"⬅️ Passage! Total: {self.cnt_in}")
                self.tracks[tid]['counted'] = True
                self.tracks[tid]['frames_since_crossing'] = 0
            # Sens inverse ignoré en mode latéral (pas de cnt_out)
        else:
            # Mode top : comptage bidirectionnel IN/OUT
            if new_center[1] > old_center[1]:
                self.cnt_in += 1
                print(f"➡️ Entrée! Total IN: {self.cnt_in}")
            else:
                self.cnt_out += 1
                print(f"⬅️ Sortie! Total OUT: {self.cnt_out}")
            self.tracks[tid]['counted'] = True
            self.tracks[tid]['frames_since_crossing'] = 0

    def _predict(self):
        for tid in self.tracks:
            if self.kalman_enabled and self.tracks[tid]['kalman'] is not None:
                predicted = self.tracks[tid]['kalman'].predict()
                cx, cy = predicted[0,0], predicted[1,0]
                w = self.tracks[tid]['bbox'][2] - self.tracks[tid]['bbox'][0]
                h = self.tracks[tid]['bbox'][3] - self.tracks[tid]['bbox'][1]
                self.tracks[tid]['predicted_bbox'] = [int(cx-w//2), int(cy-h//2), int(cx+w//2), int(cy+h//2)]

    def _associate(self, detections):
        if not self.tracks or not detections:
            return {}, list(range(len(detections))), list(self.tracks.keys())

        t_ids = list(self.tracks.keys())
        iou_matrix = np.zeros((len(t_ids), len(detections)))
        for t_idx, tid in enumerate(t_ids):
            target_box = self.tracks[tid].get('predicted_bbox', self.tracks[tid]['bbox'])
            for d_idx, det in enumerate(detections):
                iou_matrix[t_idx, d_idx] = self._compute_iou(target_box, det['bbox'])

        matched, unmatched_det = {}, list(range(len(detections)))
        unmatched_track = list(range(len(t_ids)))

        while True:
            max_val = np.max(iou_matrix) if iou_matrix.size > 0 else 0
            # si IoU insuffisant, essayer la distance centroïde
            if max_val < self.iou_threshold:
                # Fallback : associer par distance centroïde (max 80px)
                for t_idx, tid in enumerate(t_ids):
                    if t_idx not in unmatched_track:
                        continue
                    tc = self.tracks[tid]['center']
                    for d_idx in unmatched_det:
                        dc = ((detections[d_idx]['bbox'][0]+detections[d_idx]['bbox'][2])//2,
                              (detections[d_idx]['bbox'][1]+detections[d_idx]['bbox'][3])//2)
                        if np.sqrt((tc[0]-dc[0])**2+(tc[1]-dc[1])**2) < 80:
                            matched[d_idx] = tid
                            unmatched_track.remove(t_idx)
                            unmatched_det.remove(d_idx)
                            break
                break
            t_idx, d_idx = np.unravel_index(np.argmax(iou_matrix), iou_matrix.shape)
            matched[d_idx] = t_ids[t_idx]
            iou_matrix[t_idx, :] = -1
            iou_matrix[:, d_idx] = -1
            if t_idx in unmatched_track: unmatched_track.remove(t_idx)
            if d_idx in unmatched_det: unmatched_det.remove(d_idx)

        return matched, unmatched_det, [t_ids[i] for i in unmatched_track]

    def update(self, detections, frame_shape=None):
        if frame_shape is not None:
            h, w = frame_shape[0], frame_shape[1]
            if self.ligne is None or isinstance(self.ligne, int):
                if self.mode == 'lateral':
                    # Ligne verticale : position X passée ou milieu par défaut
                    x_pos = self.ligne if isinstance(self.ligne, int) else w // 2
                    self.ligne = ((x_pos, 0), (x_pos, h))
                else:
                    # Ligne horizontale (mode top) : position Y passée ou milieu
                    y_pos = self.ligne if isinstance(self.ligne, int) else h // 2
                    self.ligne = ((0, y_pos), (w, y_pos))
            self.smoother.update_shape(h, w)

        self._predict()
        matched, unmatched_det, unmatched_track = self._associate(detections)

        # Update existing tracks
        for d_idx, tid in matched.items():
            det = detections[d_idx]
            smooth_bbox, _ = self.smoother.smooth(tid, det['bbox'], det['conf'])

            # new_center calculé EN PREMIER
            new_center = ((smooth_bbox[0]+smooth_bbox[2])//2, (smooth_bbox[1]+smooth_bbox[3])//2)
            old_center = self.tracks[tid]['center']

            # Reset counted si la personne a changé de côté (demi-tour)
            # avec cooldown pour éviter le double comptage sur va-et-vient
            current_side = self._get_side(new_center)
            cooldown_elapsed = (self.tracks[tid].get('frames_since_crossing', self.crossing_cooldown)
                                >= self.crossing_cooldown)
            if (self.tracks[tid].get('last_side')
                    and self.tracks[tid]['last_side'] != current_side
                    and cooldown_elapsed):
                self.tracks[tid]['counted'] = False

            # Incrémenter le compteur de cooldown
            self.tracks[tid]['frames_since_crossing'] = self.tracks[tid].get(
                'frames_since_crossing', self.crossing_cooldown) + 1

            # Détection du franchissement
            if not self.tracks[tid]['counted']:
                if self._intersect(old_center, new_center, self.ligne[0], self.ligne[1]):
                    self._count_crossing(old_center, new_center, tid)
                    self.tracks[tid]['last_side'] = current_side

            self.tracks[tid].update({
                'bbox': smooth_bbox,
                'center': new_center,
                'lost': 0,
                'last_side': current_side
            })
            if self.kalman_enabled:
                self.tracks[tid]['kalman'].correct(
                    np.array([[new_center[0]], [new_center[1]]], np.float32)
                )

        # New detections
        for d_idx in unmatched_det:
            det = detections[d_idx]
            if det['conf'] < self.min_confidence:
                continue
            cx = (det['bbox'][0]+det['bbox'][2])//2
            cy = (det['bbox'][1]+det['bbox'][3])//2
            side = self._get_side((cx, cy))
            self.tracks[self.next_id] = {
                'bbox': det['bbox'],
                'center': (cx, cy),
                'lost': 0,
                'counted': False,
                'last_side': side,
                'kalman': self._init_kalman(det['bbox'])
            }
            self.next_id += 1

        # Cleanup lost tracks
        for tid in unmatched_track:
            self.tracks[tid]['lost'] += 1
        to_del = [tid for tid, t in self.tracks.items() if t['lost'] > self.max_lost]
        for tid in to_del:
            del self.tracks[tid]

        # Nettoyer l'historique du smoother pour les tracks supprimés
        active_tracks = set(self.tracks.keys())
        self.smoother.cleanup(active_tracks)

        return [(tid, {'bbox': t['bbox']}) for tid, t in self.tracks.items() if t['lost'] == 0]

class CompteurHailoFinal:
    """Application principale"""

    def __init__(self, hef_path, input_source=0, conf_threshold=0.5, line_y=None,
                 mode='top', frame_skip=1, count_direction='left-to-right'):
        self.hef_path = hef_path
        self.input_source = input_source
        self.conf_threshold = conf_threshold
        self.line_y = line_y  # Position Y (mode top) ou X (mode lateral)
        self.mode = mode
        self.frame_skip = max(1, frame_skip)
        self.tracker = TrajectoryTracker(mode=mode, count_direction=count_direction)
        self.postproc = YOLOPostProcess()
        self.vdevice = None
        self.network_group = None
        self.infer_pipeline = None
        self.input_vstream_info = None
        self.output_vstream_info = None
        self.is_video_file = isinstance(input_source, str) and os.path.exists(input_source)
        self.diag_frame_count = 0  # V48: compteur pour diagnostic

        if not os.path.exists(hef_path):
            raise FileNotFoundError(f"Modèle non trouvé: {hef_path}")

        self.init_hailo()

    def init_hailo(self):
        """Initialisation Hailo"""
        print(f"📁 Chargement: {self.hef_path}")
        self.hef = HEF(self.hef_path)

        # === V48 DIAGNOSTIC : lister exhaustivement TOUTES les entrées/sorties ===
        all_inputs = self.hef.get_input_vstream_infos()
        all_outputs = self.hef.get_output_vstream_infos()
        print(f"\n{'='*70}")
        print(f"MODELE: {self.hef_path}")
        print(f"  Inputs ({len(all_inputs)}):")
        for i, inp in enumerate(all_inputs):
            print(f"    [{i}] '{inp.name}' shape={inp.shape}")
        print(f"  Outputs ({len(all_outputs)}):")
        for i, out in enumerate(all_outputs):
            print(f"    [{i}] '{out.name}' shape={out.shape}")
        print(f"{'='*70}\n")
        # === FIN DIAGNOSTIC ===

        self.input_vstream_info = self.hef.get_input_vstream_infos()[0]
        self.output_vstream_info = self.hef.get_output_vstream_infos()[0]

        print(f"📥 Entrée: {self.input_vstream_info.name}, shape: {self.input_vstream_info.shape}")
        print(f"📤 Sortie: {self.output_vstream_info.name}, shape: {self.output_vstream_info.shape}")

        if len(self.input_vstream_info.shape) == 3:
            self.input_height = self.input_vstream_info.shape[0]
            self.input_width = self.input_vstream_info.shape[1]
        elif len(self.input_vstream_info.shape) == 4:
            self.input_height = self.input_vstream_info.shape[2]
            self.input_width = self.input_vstream_info.shape[3]

        print(f"📥 Dimensions d'entrée: {self.input_width}x{self.input_height}")

        try:
            params = VDevice.create_params()
            params.scheduling_algorithm = HailoSchedulingAlgorithm.NONE
            self.vdevice = VDevice(params=params)
            print("✅ VDevice créé")

            configure_params = ConfigureParams.create_from_hef(
                hef=self.hef,
                interface=HailoStreamInterface.PCIe
            )
            print("✅ Paramètres de configuration créés")

            network_groups = self.vdevice.configure(self.hef, configure_params)
            self.network_group = network_groups[0]
            self.network_group_params = self.network_group.create_params()
            print("✅ Réseau configuré")

            self.input_vstreams_params = InputVStreamParams.make(
                self.network_group,
                format_type=FormatType.UINT8
            )

            self.output_vstreams_params = OutputVStreamParams.make(
                self.network_group,
                format_type=FormatType.FLOAT32
            )

            print("✅ Paramètres des streams créés")

            self.infer_pipeline = InferVStreams(
                self.network_group,
                self.input_vstreams_params,
                self.output_vstreams_params
            )
            print("✅ Pipeline d'inférence créé")

        except Exception as e:
            print(f"❌ Erreur initialisation Hailo: {e}")
            import traceback
            traceback.print_exc()
            raise

    def preprocess_single(self, frame):
        h, w = frame.shape[:2]
        scale = min(self.input_width / w, self.input_height / h)
        new_w, new_h = int(w * scale), int(h * scale)
        resized = cv2.resize(frame, (new_w, new_h))
        pad_x = (self.input_width - new_w) // 2
        pad_y = (self.input_height - new_h) // 2
        padded = np.full((self.input_height, self.input_width, 3), 114, dtype=np.uint8)
        padded[pad_y:pad_y+new_h, pad_x:pad_x+new_w] = resized
        return np.ascontiguousarray(np.expand_dims(padded, 0)), scale, pad_x, pad_y

    def infer_tile(self, tensor, scale, pad_x, pad_y, tile_shape, offset_x, offset_y):
        input_data = {self.input_vstream_info.name: tensor}
        output_data = self.infer_pipeline.infer(input_data)

        # === V48 DIAGNOSTIC : dumper le contenu brut pour les 3 premières frames ===
        if self.diag_frame_count <= 3:
            for key, val in output_data.items():
                raw = np.array(val)
                flat = raw.flatten().astype(np.float32)
                nonzero_count = np.count_nonzero(flat)
                print(f"\n--- FRAME {self.diag_frame_count} | TILE offset=({offset_x},{offset_y}) tile_size={tile_shape} ---")
                print(f"  Output '{key}': shape_raw={raw.shape}, dtype={raw.dtype}")
                print(f"  Taille flatten: {flat.size} floats")
                print(f"  Min={flat.min():.6f}, Max={flat.max():.6f}")
                print(f"  Non-zero: {nonzero_count}/{flat.size}")

                # Afficher TOUTES les valeurs non-nulles groupées par 5
                if nonzero_count > 0:
                    # Trouver la fin des données utiles (après le dernier non-zero)
                    nonzero_indices = np.nonzero(flat)[0]
                    last_nonzero = nonzero_indices[-1] if len(nonzero_indices) > 0 else 0
                    useful_size = last_nonzero + 1
                    n_dets = useful_size // 5
                    remainder = useful_size % 5
                    print(f"  → Données utiles: {useful_size} floats = {n_dets} détections × 5 (reste {remainder})")
                    for d in range(min(n_dets, 20)):
                        chunk = flat[d*5:(d+1)*5]
                        if np.any(chunk != 0):
                            # Interprétation [ymin, xmin, ymax, xmax, score] (normalisé)
                            y1_px = ((chunk[0] * self.input_height) - pad_y) / scale + offset_y
                            x1_px = ((chunk[1] * self.input_width) - pad_x) / scale + offset_x
                            y2_px = ((chunk[2] * self.input_height) - pad_y) / scale + offset_y
                            x2_px = ((chunk[3] * self.input_width) - pad_x) / scale + offset_x
                            score = chunk[4]
                            w_px = x2_px - x1_px
                            h_px = y2_px - y1_px
                            print(f"    det[{d}]: raw=[{chunk[0]:.4f}, {chunk[1]:.4f}, {chunk[2]:.4f}, {chunk[3]:.4f}, {chunk[4]:.4f}]")
                            print(f"            interp[ymin,xmin,ymax,xmax,score]: pixels=[x1={x1_px:.0f}, y1={y1_px:.0f}, x2={x2_px:.0f}, y2={y2_px:.0f}] ({w_px:.0f}x{h_px:.0f}) score={score:.4f}")
                            # Aussi tester l'interprétation alternative [xmin, ymin, xmax, ymax, score]
                            x1_alt = ((chunk[0] * self.input_width) - pad_x) / scale + offset_x
                            y1_alt = ((chunk[1] * self.input_height) - pad_y) / scale + offset_y
                            x2_alt = ((chunk[2] * self.input_width) - pad_x) / scale + offset_x
                            y2_alt = ((chunk[3] * self.input_height) - pad_y) / scale + offset_y
                            w_alt = x2_alt - x1_alt
                            h_alt = y2_alt - y1_alt
                            print(f"            interp[xmin,ymin,xmax,ymax,score]: pixels=[x1={x1_alt:.0f}, y1={y1_alt:.0f}, x2={x2_alt:.0f}, y2={y2_alt:.0f}] ({w_alt:.0f}x{h_alt:.0f}) score={score:.4f}")
                else:
                    print(f"  → AUCUNE valeur non-nulle (tile vide)")

                # V48: aussi tester avec conf_threshold=0.1
                postproc_diag = YOLOPostProcess(elements_per_det=self.postproc.elements_per_det)
                postproc_diag.debug_done = True  # Pas de debug supplémentaire
                dets_low_conf = postproc_diag.process_yolo_output(
                    val,
                    conf_threshold=0.1,
                    original_shape=tile_shape,
                    input_shape=(self.input_height, self.input_width),
                    scale=scale, pad_x=pad_x, pad_y=pad_y
                )
                print(f"  Avec conf=0.1: {len(dets_low_conf)} détections")
                for d in dets_low_conf:
                    bx = d['bbox']
                    print(f"    bbox=[{bx[0]},{bx[1]},{bx[2]},{bx[3]}] conf={d['conf']:.4f} size={bx[2]-bx[0]}x{bx[3]-bx[1]}")
        # === FIN DIAGNOSTIC ===

        dets = []
        if output_data and self.output_vstream_info.name in output_data:
            dets = self.postproc.process_yolo_output(
                output_data[self.output_vstream_info.name],
                conf_threshold=self.conf_threshold,
                original_shape=tile_shape,
                input_shape=(self.input_height, self.input_width),
                scale=scale, pad_x=pad_x, pad_y=pad_y
            )
            for d in dets:
                b = d['bbox']
                d['bbox'] = [b[0]+offset_x, b[1]+offset_y, b[2]+offset_x, b[3]+offset_y]
                d['center_y'] = (d['bbox'][1] + d['bbox'][3]) // 2
        return dets

    def detect_tiled(self, frame):
        h, w = frame.shape[:2]
        cols, rows = (3, 2) if w >= 1280 else (2, 2)
        overlap = 0.15
        tw, th = w // cols, h // rows
        pw, ph = int(tw * overlap), int(th * overlap)
        all_dets = []
        for row in range(rows):
            for col in range(cols):
                x1 = max(0, col * tw - pw)
                y1 = max(0, row * th - ph)
                x2 = min(w, (col+1) * tw + pw)
                y2 = min(h, (row+1) * th + ph)
                tile = frame[y1:y2, x1:x2]
                tensor, scale, px, py = self.preprocess_single(tile)
                dets = self.infer_tile(tensor, scale, px, py, tile.shape[:2], x1, y1)
                all_dets.extend(dets)
        return self.nms(all_dets)

    def nms(self, detections, iou_threshold=0.4):
        if not detections:
            return []
        boxes = np.array([d['bbox'] for d in detections], dtype=float)
        scores = np.array([d['conf'] for d in detections])
        x1,y1,x2,y2 = boxes[:,0],boxes[:,1],boxes[:,2],boxes[:,3]
        areas = (x2-x1)*(y2-y1)
        order = scores.argsort()[::-1]
        keep = []
        while order.size > 0:
            i = order[0]; keep.append(i)
            ix1 = np.maximum(x1[i], x1[order[1:]])
            iy1 = np.maximum(y1[i], y1[order[1:]])
            ix2 = np.minimum(x2[i], x2[order[1:]])
            iy2 = np.minimum(y2[i], y2[order[1:]])
            inter = np.maximum(0,ix2-ix1)*np.maximum(0,iy2-iy1)
            iou = inter/(areas[i]+areas[order[1:]]-inter+1e-6)
            order = order[1:][iou < iou_threshold]
        return [detections[i] for i in keep]

    def run(self):
        """Boucle principale"""
        if self.is_video_file:
            cap = cv2.VideoCapture(self.input_source)
            source_type = "fichier vidéo"
        else:
            cap = cv2.VideoCapture(self.input_source)
            source_type = "caméra USB"

        if not cap.isOpened():
            print(f"❌ Erreur: impossible d'ouvrir la source {self.input_source}")
            return

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps_source = cap.get(cv2.CAP_PROP_FPS)

        print(f"📹 Source: {source_type}")
        print(f"📐 Résolution: {width}x{height}")

        if self.is_video_file and fps_source > 0:
            print(f"⏱️ FPS vidéo: {fps_source:.2f}")

        if self.is_video_file:
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if total_frames <= 0:
                total_frames = 1  # Éviter division par zéro
            print(f"📊 Frames totales: {total_frames}")

        if self.line_y is None:
            if self.mode == 'lateral':
                self.tracker.ligne = width // 2
            else:
                self.tracker.ligne = height // 2
        else:
            self.tracker.ligne = self.line_y

        if self.mode == 'lateral':
            print(f"📏 Ligne de comptage verticale: X={self.tracker.ligne}")
            print(f"🚶 Direction comptée: {self.tracker.count_direction}")
        else:
            print(f"📏 Ligne de comptage horizontale: Y={self.tracker.ligne}")

        if self.frame_skip > 1:
            print(f"⏩ Frame skip: traitement 1 frame sur {self.frame_skip}")

        print(f"📐 Mode: {self.mode}")
        print("🚀 Démarrage du compteur...")
        print("Appuyez sur 'q' pour quitter")

        frame_count = 0
        fps_time = time.time()

        output_video = None
        if self.is_video_file:
            input_path = Path(self.input_source)
            output_path = input_path.parent / f"{input_path.stem}_compte{input_path.suffix}"
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out_fps = fps_source if fps_source > 0 else 30.0
            output_video = cv2.VideoWriter(str(output_path), fourcc, out_fps, (width, height))
            print(f"💾 Sauvegarde vidéo: {output_path}")

        try:
            with self.network_group.activate(self.network_group_params):
                with self.infer_pipeline:
                    detections = []
                    tracked = []
                    while True:
                        ret, frame = cap.read()
                        if not ret:
                            print("🏁 Fin de la vidéo")
                            break

                        frame_count += 1
                        self.diag_frame_count = frame_count  # V48: accessible depuis infer_tile

                        # Frame skipping : on lit toutes les frames mais on n'infère que 1/N
                        # Les frames sautées réutilisent les dernières détections
                        skip_this = (self.frame_skip > 1 and frame_count % self.frame_skip != 1)

                        if self.is_video_file and frame_count % 30 == 0:
                            progress = (frame_count / total_frames) * 100
                            print(f"  Progression: {frame_count}/{total_frames} ({progress:.1f}%)")

                        # === V48 DIAGNOSTIC : full-frame test pour frame 1 ===
                        if frame_count == 1 and not skip_this:
                            print(f"\n{'='*70}")
                            print("DIAGNOSTIC FRAME 1 - FULL FRAME (sans tiling)")
                            print(f"  Image: {width}x{height}")
                            print(f"{'='*70}")
                            tensor_full, scale_f, px_f, py_f = self.preprocess_single(frame)
                            print(f"  Scale={scale_f:.4f}, pad=({px_f},{py_f})")
                            # infer_tile va imprimer le diagnostic car diag_frame_count=1
                            dets_full = self.infer_tile(tensor_full, scale_f, px_f, py_f, (height, width), 0, 0)
                            print(f"\n  Détections full-frame (conf>={self.conf_threshold}): {len(dets_full)}")
                            for d in dets_full:
                                bx = d['bbox']
                                print(f"    bbox=[{bx[0]},{bx[1]},{bx[2]},{bx[3]}] conf={d['conf']:.4f} size={bx[2]-bx[0]}x{bx[3]-bx[1]}")
                            print(f"{'='*70}\n")

                            # Aussi afficher l'entête du tiling
                            cols, rows = (3, 2) if width >= 1280 else (2, 2)
                            print(f"DIAGNOSTIC FRAME 1 - TILING {cols}x{rows}")
                            print(f"{'='*70}")
                        # === FIN DIAGNOSTIC FULL FRAME ===

                        if not skip_this:
                            detections = self.detect_tiled(frame)
                            tracked = self.tracker.update(detections, frame_shape=(height, width))

                        # === V48: résumé après frame 3 ===
                        if frame_count == 3:
                            print(f"\n{'='*70}")
                            print("FIN DU DIAGNOSTIC (frames 1-3)")
                            print(f"  Détections frame 3: {len(detections)}")
                            print(f"  Tracks actifs: {len(tracked)}")
                            print(f"  IDs attribués jusqu'ici: {self.tracker.next_id}")
                            print(f"{'='*70}")
                            print("\n📋 Info pour debug manuel :")
                            print("  Pour comparer, lancer l'exemple officiel Hailo :")
                            print("  python ~/hailo-rpi5-examples/basic_pipelines/detection.py --input Marche2_P.mp4")
                            print("  Si l'exemple officiel détecte bien les personnes, le problème est dans notre post-processing.")
                            print(f"{'='*70}\n")

                        for track_id, det in tracked:
                            bbox = det['bbox']
                            is_counted = self.tracker.tracks.get(track_id, {}).get('counted', False)
                            color = (0, 0, 255) if is_counted else (0, 255, 0)
                            label_color = (255, 100, 0) if is_counted else (255, 255, 0)

                            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
                            cv2.putText(frame, f"ID:{track_id}", (bbox[0], bbox[1]-10),
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, label_color, 1)

                        # Ligne de comptage
                        cv2.line(frame, self.tracker.ligne[0], self.tracker.ligne[1], (0, 255, 255), 3)

                        # Dashboard semi-transparent
                        overlay = frame.copy()
                        if self.mode == 'lateral':
                            # Mode latéral : affichage simplifié (entrées uniquement)
                            cv2.rectangle(overlay, (5, 5), (220, 70), (0, 0, 0), -1)
                            cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
                            cv2.putText(frame, f"PASSAGES: {self.tracker.cnt_in}", (15, 40),
                                        cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 255, 0), 2)
                            # Indicateur de direction
                            dir_text = ">>>" if self.tracker.count_direction == 'left-to-right' else "<<<"
                            cv2.putText(frame, dir_text, (15, 62),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                        else:
                            # Mode top : affichage IN/OUT
                            cv2.rectangle(overlay, (5, 5), (220, 110), (0, 0, 0), -1)
                            cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
                            cv2.putText(frame, f"ENTREES: {self.tracker.cnt_in}", (15, 40),
                                        cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 255, 0), 2)
                            cv2.putText(frame, f"SORTIES: {self.tracker.cnt_out}", (15, 85),
                                        cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 0, 255), 2)

                        if self.is_video_file:
                            cv2.putText(frame, f"Frame: {frame_count}/{total_frames}", (10, 125),
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                        else:
                            if frame_count % 30 == 0:
                                now = time.time()
                                fps = 30 / (now - fps_time)
                                fps_time = now
                                cv2.putText(frame, f"FPS: {fps:.1f}", (10, 90),
                                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)

                        cv2.putText(frame, f"Détections: {len(detections)}", (10, 130),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

                        cv2.imshow("Compteur Hailo", frame)

                        if output_video is not None:
                            output_video.write(frame)

                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            break
        finally:
            cap.release()
            if output_video is not None:
                output_video.release()
                print(f"✅ Vidéo sauvegardée")
            cv2.destroyAllWindows()
            if self.vdevice:
                self.vdevice.release()

        self.sauvegarder_compte()
        print(f"entrees: {self.tracker.cnt_in}")
        if self.mode == 'top':
            print(f"sorties: {self.tracker.cnt_out}")

    def sauvegarder_compte(self):
        data = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "mode": self.mode,
            "entrees": self.tracker.cnt_in,
            "sorties": self.tracker.cnt_out if self.mode == 'top' else None,
            "total_passage": self.tracker.cnt_in + self.tracker.cnt_out,
            "modele": os.path.basename(self.hef_path),
            "source": str(self.input_source),
            "conf_threshold": self.conf_threshold,
            "frame_skip": self.frame_skip
        }
        try:
            with open("compteur_stats.json", "w") as f:
                json.dump(data, f, indent=4)
        except Exception as e:
            print(f"Erreur sauvegarde: {e}")

def trouver_hef_detection():
    base_paths = [
        "/usr/local/hailo/resources/models/hailo8",
        "/usr/local/hailo/resources/models/hailo8l",
        str(Path.home() / "hailo-rpi5-examples/resources"),
        "."
    ]

    print("🔍 Recherche d'un modèle de détection...")

    model_preferences = [
        "yolov8s.hef",
        "yolov8n.hef",
        "yolov5m.hef",
        "yolov5s.hef",
        "yolov7.hef",
        "yolov6n.hef"
    ]

    for base in base_paths:
        if os.path.exists(base):
            for model in model_preferences:
                full_path = os.path.join(base, model)
                if os.path.exists(full_path):
                    print(f"✅ Modèle trouvé: {full_path}")
                    return full_path

    return None

def main():
    parser = argparse.ArgumentParser(description="Compteur de personnes avec Hailo")
    parser.add_argument('--input', '-i', default='0',
                       help='Source d\'entrée: 0 pour caméra USB, ou chemin vers fichier vidéo')
    parser.add_argument('--conf', '-c', type=float, default=0.5,
                       help='Seuil de confiance pour les détections (défaut: 0.5)')
    parser.add_argument('--line', '-l', type=int, default=None,
                       help='Position de la ligne de comptage: Y en mode top, X en mode lateral (défaut: milieu)')
    parser.add_argument('--mode', '-m', choices=['top', 'lateral'], default='top',
                       help='Mode de vue: top (plongeante, IN/OUT) ou lateral (côté, unidirectionnel)')
    parser.add_argument('--skip', '-s', type=int, default=1,
                       help='Frame skipping: traiter 1 frame sur N (défaut: 1 = toutes)')
    parser.add_argument('--direction', '-d', choices=['left-to-right', 'right-to-left'],
                       default='left-to-right',
                       help='Direction d\'entrée en mode lateral (défaut: left-to-right)')

    args = parser.parse_args()

    if args.line is not None and args.line < 0:
        print(f"❌ --line doit être >= 0 (reçu: {args.line})")
        return

    if args.skip < 1:
        print(f"❌ --skip doit être >= 1 (reçu: {args.skip})")
        return

    print(f"📦 Numpy version: {np.__version__}")
    print(f"🎯 Seuil de confiance: {args.conf}")
    print(f"📐 Mode: {args.mode}")
    if args.mode == 'lateral':
        print(f"🚶 Direction: {args.direction}")
    if args.skip > 1:
        print(f"⏩ Frame skip: 1/{args.skip}")

    if args.input == '0' or args.input.lower() == 'usb':
        input_source = 0
        print("📹 Source: Caméra USB")
    else:
        input_source = args.input
        if os.path.exists(input_source):
            print(f"📹 Source: Fichier vidéo {input_source}")
        else:
            print(f"❌ Fichier vidéo non trouvé: {input_source}")
            return

    hef_path = trouver_hef_detection()

    if not hef_path:
        print("\n❌ Aucun modèle trouvé!")
        return

    try:
        app = CompteurHailoFinal(hef_path, input_source, args.conf, args.line,
                                 mode=args.mode, frame_skip=args.skip,
                                 count_direction=args.direction)
        app.run()
    except KeyboardInterrupt:
        print("\n⏹️ Arrêt demandé")
    except Exception as e:
        print(f"❌ Erreur: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
