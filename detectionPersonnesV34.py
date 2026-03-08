#!/usr/bin/env python3
"""
compteur_hailo_v38.py

Format de sortie Hailo NMS BY CLASS (FLOAT32) pour yolov8n.hef :
  Buffer max 2004 bytes = 501 floats, structuré par classe :
  [num_detections, y_min_0, x_min_0, y_max_0, x_max_0, score_0, ...]
  - Premier float = nombre de détections (cast en int)
  - Chaque détection = 5 floats : y_min, x_min, y_max, x_max, score
  - Coordonnées normalisées [0,1] relatives à l'image d'entrée 320x320
  - NMS hardware (IoU 0.70, score threshold 0.200), 1 classe (personne)
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
    """Post-processing pour yolov8n.hef - Format concaténé sans compteur"""

    def __init__(self):
        self.debug_done = False

    def process_yolo_output(self, output, conf_threshold=0.5,
                       input_shape=(320,320), original_shape=(960,540),
                       scale=1.0, pad_x=0, pad_y=0):
        """Parse le buffer de sortie (détections concaténées)"""
        detections = []

        if output is None:
            return detections

        try:
            # Convertir en ndarray et aplatir en 1D
            raw = np.array(output).flatten().astype(np.float32)

            if raw.size == 0:
                return detections

            if not self.debug_done:
                print(f"\n🔍 DEBUG - Buffer brut: {raw.shape[0]} floats")
                print(f"  Scale: {scale:.3f}, Pad: ({pad_x}, {pad_y})")
                print(f"  raw: {raw}")
                print(f"  raw[0] = {raw[0]:.3f}")

            # Format: toutes les détections concaténées, chaque détection = 5 floats
            # [y_min, x_min, y_max, x_max, score] répété
            num_detections = raw.shape[0] // 5
            
            if not self.debug_done:
                print(f"  Nombre de détections (déduit): {num_detections}")

            # Chaque détection = 5 floats : y_min, x_min, y_max, x_max, score
            for i in range(num_detections):
                offset = i * 5
                
                y_min_n = float(raw[offset + 0])
                x_min_n = float(raw[offset + 1])
                y_max_n = float(raw[offset + 2])
                x_max_n = float(raw[offset + 3])
                score   = float(raw[offset + 4])

                if score < conf_threshold:
                    continue

                # Coordonnées normalisées → pixels dans l'espace paddé
                x1p = x_min_n * input_shape[1]
                y1p = y_min_n * input_shape[0]
                x2p = x_max_n * input_shape[1]
                y2p = y_max_n * input_shape[0]

                # Enlever le padding
                x1p -= pad_x
                y1p -= pad_y
                x2p -= pad_x
                y2p -= pad_y

                # Appliquer l'échelle inverse
                x1 = int(x1p / scale)
                y1 = int(y1p / scale)
                x2 = int(x2p / scale)
                y2 = int(y2p / scale)

                # Clamp aux dimensions originales
                x1 = max(0, min(x1, original_shape[1] - 1))
                y1 = max(0, min(y1, original_shape[0] - 1))
                x2 = max(0, min(x2, original_shape[1] - 1))
                y2 = max(0, min(y2, original_shape[0] - 1))

                if not self.debug_done and i == 0:
                    print(f"\n  Première détection:")
                    print(f"    Normalisé (y,x): [{y_min_n:.3f}, {x_min_n:.3f}, {y_max_n:.3f}, {x_max_n:.3f}]")
                    print(f"    Pixels paddés: [{x1p+pad_x:.1f}, {y1p+pad_y:.1f}, {x2p+pad_x:.1f}, {y2p+pad_y:.1f}]")
                    print(f"    Final: [{x1}, {y1}, {x2}, {y2}], score: {score:.3f}")

                w = x2 - x1
                h = y2 - y1
                if w > 10 and h > 20:
                    detections.append({
                        'bbox': [x1, y1, x2, y2],
                        'conf': score,
                        'center_y': (y1 + y2) // 2
                    })

        except Exception as e:
            print(f"⚠️ Erreur post-processing: {e}")
            import traceback
            traceback.print_exc()

        if not self.debug_done:
            print(f"\n  📊 {len(detections)} détection(s) valides")
            self.debug_done = True

        return detections

class TemporalSmoother:
    """Lisse les détections sur plusieurs frames pour stabiliser les trajectoires"""
    
    def __init__(self, history_size=5, original_shape=(540, 960)):
        """
        Args:
            history_size: nombre de frames à garder en mémoire pour le lissage
            original_shape: dimensions de l'image (hauteur, largeur) pour validation
        """
        self.history = {}  # track_id -> liste de (bbox, conf)
        self.history_size = history_size
        self.original_shape = original_shape  # (height, width)
    
    def update_shape(self, height, width):
        """Met à jour les dimensions de l'image (appelé à chaque frame si besoin)"""
        self.original_shape = (height, width)
    
    def smooth(self, track_id, bbox, conf):
        """
        Ajoute une nouvelle détection et retourne la version lissée
        
        Args:
            track_id: identifiant unique de la piste
            bbox: [x1, y1, x2, y2]
            conf: score de confiance
            
        Returns:
            tuple: (bbox_lissé, conf_lissée)
        """
        # Initialiser l'historique pour ce track si nécessaire
        if track_id not in self.history:
            self.history[track_id] = []
        
        # Ajouter la nouvelle détection
        self.history[track_id].append((bbox, conf))
        
        # Limiter la taille de l'historique
        if len(self.history[track_id]) > self.history_size:
            self.history[track_id].pop(0)
        
        # Si on a assez d'historique, on lisse
        if len(self.history[track_id]) >= 3:
            # Prendre les 3 dernières détections
            recent = self.history[track_id][-3:]
            
            # Moyenne pondérée par la confiance
            total_conf = sum(b[1] for b in recent)
            if total_conf > 0:
                # Moyenne des coordonnées pondérée par la confiance
                avg_bbox = [
                    int(sum(b[0][0] * b[1] for b in recent) / total_conf),  # x1
                    int(sum(b[0][1] * b[1] for b in recent) / total_conf),  # y1
                    int(sum(b[0][2] * b[1] for b in recent) / total_conf),  # x2
                    int(sum(b[0][3] * b[1] for b in recent) / total_conf)   # y2
                ]
                avg_conf = total_conf / len(recent)
            else:
                # Fallback si confiances nulles
                avg_bbox = [
                    int(sum(b[0][0] for b in recent) / len(recent)),
                    int(sum(b[0][1] for b in recent) / len(recent)),
                    int(sum(b[0][2] for b in recent) / len(recent)),
                    int(sum(b[0][3] for b in recent) / len(recent))
                ]
                avg_conf = conf
            
            # Vérifier que la boîte lissée a une taille raisonnable
            w = avg_bbox[2] - avg_bbox[0]
            h = avg_bbox[3] - avg_bbox[1]
            
            # Si la boîte lissée est trop déformée ou hors limites, on garde l'originale
            if (w < 10 or h < 20 or 
                w > self.original_shape[1] or 
                h > self.original_shape[0] or
                avg_bbox[0] < 0 or avg_bbox[1] < 0 or
                avg_bbox[2] > self.original_shape[1] or
                avg_bbox[3] > self.original_shape[0]):
                return bbox, conf
            
            return avg_bbox, avg_conf
        else:
            # Pas assez d'historique, on retourne la détection brute
            return bbox, conf
    
    def cleanup(self, active_tracks):
        """
        Nettoie l'historique des tracks qui ne sont plus actifs
        
        Args:
            active_tracks: liste des IDs de tracks actuellement suivis
        """
        to_remove = []
        for track_id in self.history:
            if track_id not in active_tracks:
                to_remove.append(track_id)
        
        for track_id in to_remove:
            del self.history[track_id]

class TrajectoryTracker:
    """Tracker ultra-robuste avec filtrage temporel et Kalman"""
    
    def __init__(self, ligne_centrale=300):
        self.ligne = ligne_centrale
        self.compteur = 0
        self.next_id = 0
        self.tracks = {}
        
        # Initialiser le smoother (les dimensions seront mises à jour à la première frame)
        self.smoother = TemporalSmoother(history_size=5, original_shape=(540, 960))
        
        # Paramètres optimisés
        self.iou_threshold = 0.3
        self.max_lost = 8
        self.kalman_enabled = True
        self.min_confidence = 0.3
    
    def update(self, detections, frame_shape=None):
        """
        Met à jour le tracking avec les nouvelles détections
        
        Args:
            detections: liste des détections de la frame courante
            frame_shape: (hauteur, largeur) de la frame courante
        """
        # Mettre à jour les dimensions de l'image si fournies
        if frame_shape is not None:
            self.smoother.update_shape(frame_shape[0], frame_shape[1])
        
        # ... (suite du code existant)
        
        # Mettre à jour la résolution de l'image si disponible
        if detections and 'original_shape' in detections[0]:
            self.original_shape = detections[0]['original_shape']
        
        # Associer détections ↔ tracks
        matched, unmatched_det, unmatched_track = self._associate(detections)
        
        # Mettre à jour les tracks associés
        for det_idx, track_id in matched.items():
            det = detections[det_idx]
            
            # LISSAGE TEMPOREL - Application
            smooth_bbox, smooth_conf = self.smoother.smooth(
                track_id, det['bbox'], det['conf']
            )
            
            # Si la confiance lissée est trop basse, on garde quand même ?
            # On utilise le max des deux confiances pour être conservateur
            effective_conf = max(smooth_conf, det['conf'])
            if effective_conf < self.min_confidence:
                continue
            
            cx = (smooth_bbox[0] + smooth_bbox[2]) // 2
            cy = (smooth_bbox[1] + smooth_bbox[3]) // 2
            
            # Vérifier le franchissement de ligne
            old_side = self.tracks[track_id]['side']
            new_side = 'below' if cy > self.ligne else 'above'
            
            if old_side != new_side and not self.tracks[track_id]['counted']:
                self.compteur += 1
                self.tracks[track_id]['counted'] = True
                print(f"✅ Personne {track_id} comptée! Total: {self.compteur}")
            
            # Mettre à jour les données
            self.tracks[track_id].update({
                'bbox': smooth_bbox,  # Utiliser la version lissée
                'center': (cx, cy),
                'side': new_side,
                'lost': 0,
                'last_seen': time.time(),
                'confidence': effective_conf
            })
            
            # Mettre à jour Kalman
            self._update_kalman(track_id, (cx, cy))
        
        # Créer nouveaux tracks
        for det_idx in unmatched_det:
            det = detections[det_idx]
            if det['conf'] < self.min_confidence:
                continue
                
            cx = (det['bbox'][0] + det['bbox'][2]) // 2
            cy = (det['bbox'][1] + det['bbox'][3]) // 2
            
            self.tracks[self.next_id] = {
                'bbox': det['bbox'],
                'center': (cx, cy),
                'side': 'below' if cy > self.ligne else 'above',
                'lost': 0,
                'counted': False,
                'last_seen': time.time(),
                'confidence': det['conf'],
                'kalman': self._init_kalman(det['bbox'])
            }
            
            # Initialiser le smoother pour ce nouveau track
            self.smoother.smooth(self.next_id, det['bbox'], det['conf'])
            self.next_id += 1
        
        # Nettoyer les tracks perdus
        to_delete = []
        for track_id, track in self.tracks.items():
            if track['lost'] > self.max_lost:
                to_delete.append(track_id)
        
        # Nettoyer le smoother avant de supprimer les tracks
        active_tracks = set(self.tracks.keys()) - set(to_delete)
        self.smoother.cleanup(active_tracks)
        
        for tid in to_delete:
            del self.tracks[tid]
        
        # Retourner les tracks actifs
        current_tracks = []
        for track_id, track in self.tracks.items():
            if track['lost'] == 0:
                # Utiliser la bbox Kalman si disponible, sinon la bbox lissée
                if 'kalman_bbox' in track:
                    bbox = track['kalman_bbox']
                else:
                    bbox = track['bbox']
                det = {'bbox': bbox, 'center_y': track['center'][1]}
                current_tracks.append((track_id, det))
        
        return current_tracks
    
class CompteurHailoFinal:
    """Application principale"""

    def __init__(self, hef_path, input_source=0, conf_threshold=0.5, line_y=None):
        self.hef_path = hef_path
        self.input_source = input_source
        self.conf_threshold = conf_threshold
        self.line_y = line_y
        self.tracker = TrajectoryTracker()
        self.postproc = YOLOPostProcess()
        self.vdevice = None
        self.network_group = None
        self.infer_pipeline = None
        self.input_vstream_info = None
        self.output_vstream_info = None
        self.is_video_file = isinstance(input_source, str) and os.path.exists(input_source)

        if not os.path.exists(hef_path):
            raise FileNotFoundError(f"Modèle non trouvé: {hef_path}")

        self.init_hailo()

    def init_hailo(self):
        """Initialisation Hailo"""
        print(f"📁 Chargement: {self.hef_path}")
        self.hef = HEF(self.hef_path)

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
        """Prétraitement standard (utilisé pour chaque tuile)"""
        h, w = frame.shape[:2]
        scale = min(self.input_width / w, self.input_height / h)
        new_w = int(w * scale)
        new_h = int(h * scale)
        resized = cv2.resize(frame, (new_w, new_h))
        pad_x = (self.input_width - new_w) // 2
        pad_y = (self.input_height - new_h) // 2
        padded = np.full((self.input_height, self.input_width, 3), 114, dtype=np.uint8)
        padded[pad_y:pad_y+new_h, pad_x:pad_x+new_w] = resized
        tensor = np.expand_dims(padded, axis=0)
        return np.ascontiguousarray(tensor), scale, pad_x, pad_y

    def infer_and_detect(self, tensor, scale, pad_x, pad_y, original_shape, offset_x=0, offset_y=0):
        """Inférence sur un tenseur et retourne les détections avec offset appliqué"""
        input_data = {self.input_vstream_info.name: tensor}
        output_data = self.infer_pipeline.infer(input_data)

        detections = []
        if output_data and self.output_vstream_info.name in output_data:
            model_output = output_data[self.output_vstream_info.name]
            dets = self.postproc.process_yolo_output(
                model_output,
                conf_threshold=self.conf_threshold,
                original_shape=original_shape,
                input_shape=(self.input_height, self.input_width),
                scale=scale,
                pad_x=pad_x,
                pad_y=pad_y
            )
            # Appliquer l'offset de la tuile
            for d in dets:
                b = d['bbox']
                d['bbox'] = [b[0]+offset_x, b[1]+offset_y, b[2]+offset_x, b[3]+offset_y]
                d['center_y'] = (d['bbox'][1] + d['bbox'][3]) // 2
                detections.append(d)
        return detections

    def detect_tiled(self, frame):
        """Découpe le frame en 4 tuiles qui se chevauchent et fusionne les détections"""
        h, w = frame.shape[:2]
        all_detections = []

        # 4 tuiles avec 20% de chevauchement
        overlap = 0.20
        tiles = [
            (0,        0,        w//2 + int(w*overlap/2),  h//2 + int(h*overlap/2)),   # haut-gauche
            (w//2 - int(w*overlap/2), 0, w,                h//2 + int(h*overlap/2)),   # haut-droite
            (0,        h//2 - int(h*overlap/2), w//2 + int(w*overlap/2), h),           # bas-gauche
            (w//2 - int(w*overlap/2), h//2 - int(h*overlap/2), w, h),                  # bas-droite
        ]

        for (x1, y1, x2, y2) in tiles:
            tile = frame[y1:y2, x1:x2]
            tile_h, tile_w = tile.shape[:2]
            tensor, scale, pad_x, pad_y = self.preprocess_single(tile)
            dets = self.infer_and_detect(
                tensor, scale, pad_x, pad_y,
                original_shape=(tile_h, tile_w),
                offset_x=x1, offset_y=y1
            )
            all_detections.extend(dets)

        # NMS global pour supprimer les doublons dans les zones de chevauchement
        return self.nms(all_detections, iou_threshold=0.4)

    def nms(self, detections, iou_threshold=0.4):
        """Non-Maximum Suppression pour supprimer les boîtes dupliquées"""
        if not detections:
            return []

        boxes = np.array([d['bbox'] for d in detections], dtype=float)
        scores = np.array([d['conf'] for d in detections])

        x1, y1, x2, y2 = boxes[:,0], boxes[:,1], boxes[:,2], boxes[:,3]
        areas = (x2 - x1) * (y2 - y1)
        order = scores.argsort()[::-1]

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            ix1 = np.maximum(x1[i], x1[order[1:]])
            iy1 = np.maximum(y1[i], y1[order[1:]])
            ix2 = np.minimum(x2[i], x2[order[1:]])
            iy2 = np.minimum(y2[i], y2[order[1:]])
            inter = np.maximum(0, ix2-ix1) * np.maximum(0, iy2-iy1)
            iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
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
            print(f"⏱️  FPS vidéo: {fps_source:.2f}")

        if self.is_video_file:
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            print(f"📊 Frames totales: {total_frames}")

        # self.tracker.ligne = height // 1.5
        if self.line_y is None:
            self.tracker.ligne = int(height * 0.67)  # ligne à 2/3 de l'image = zone de passage
        else:
            self.tracker.ligne = self.line_y
        print(f"📏 Ligne de comptage: Y={self.tracker.ligne}")

        print("🚀 Démarrage du compteur...")
        print("Appuyez sur 'q' pour quitter")

        frame_count = 0
        fps_time = time.time()

        output_video = None
        if self.is_video_file:
            input_path = Path(self.input_source)
            output_path = input_path.parent / f"{input_path.stem}_compte{input_path.suffix}"
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            output_video = cv2.VideoWriter(str(output_path), fourcc, 30.0, (width, height))
            print(f"💾 Sauvegarde vidéo: {output_path}")

        with self.network_group.activate(self.network_group_params):
            with self.infer_pipeline:
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        print("🏁 Fin de la vidéo")
                        break

                    frame_count += 1

                    if self.is_video_file and frame_count % 30 == 0:
                        progress = (frame_count / total_frames) * 100
                        print(f"  Progression: {frame_count}/{total_frames} ({progress:.1f}%)")

                    detections = self.detect_tiled(frame)

                    tracked = self.tracker.update(detections)

                    # Dessin
                    for track_id, det in tracked:
                        bbox = det['bbox']
                        cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
                        cv2.putText(frame, f"ID:{track_id}", (bbox[0], bbox[1]-10),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

                    # cv2.line(frame, (0, self.tracker.ligne), (frame.shape[1], self.tracker.ligne), (0, 255, 0), 2)
                    cv2.line(frame, (0, int(self.tracker.ligne)), (frame.shape[1], int(self.tracker.ligne)), (0, 255, 0), 2)

                    cv2.putText(frame, f"Compteur: {self.tracker.compteur}", (10, 50),
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                    if self.is_video_file:
                        cv2.putText(frame, f"Frame: {frame_count}/{total_frames}", (10, 90),
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

        cap.release()
        if output_video is not None:
            output_video.release()
            print(f"✅ Vidéo sauvegardée")
        cv2.destroyAllWindows()
        if self.vdevice:
            self.vdevice.release()

        self.sauvegarder_compte()
        print(f"💾 Compteur final: {self.tracker.compteur}")

    def sauvegarder_compte(self):
        data = {
            "personnes": self.tracker.compteur,
            "timestamp": time.time(),
            "modele": os.path.basename(self.hef_path),
            "source": str(self.input_source),
            "conf_threshold": self.conf_threshold
        }
        with open("compte_personnes.json", 'w') as f:
            json.dump(data, f, indent=4)

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
    parser.add_argument('--line', '-l', type=int, default=None,
                       help='Position Y de la ligne de comptage (défaut: milieu de l\'image)')
    parser.add_argument('--iou', type=float, default=0.4,
                   help='Seuil IoU pour NMS (défaut: 0.4)')
    parser.add_argument('--conf', '-c', type=float, default=0.3,
                    help='Seuil de confiance (défaut: 0.3)')
    parser.add_argument('--kalman', action='store_true',
                    help='Activer le filtre de Kalman')

    args = parser.parse_args()

    print(f"📦 Numpy version: {np.__version__}")
    print(f"🎯 Seuil de confiance: {args.conf}")

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
        app = CompteurHailoFinal(hef_path, input_source, args.conf, args.line)
        app.run()
    except KeyboardInterrupt:
        print("\n⏹️ Arrêt demandé")
    except Exception as e:
        print(f"❌ Erreur: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
