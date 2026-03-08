#!/usr/bin/env python3
"""
compteur_hailo_final.py - Version complète et fonctionnelle
Compteur de personnes avec HailoRT 4.20.0 sur Raspberry Pi 5
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
    """Post-processing pour yolov8n.hef - format concaténé sans compteur"""
    
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
                print(f"  raw[0] = {raw[0]:.3f}")
            
            # Format: toutes les détections concaténées, chaque détection = 5 floats
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
    """Tracker robuste avec IOU, Kalman et lissage temporel"""
    
    def __init__(self, ligne_centrale=300):
        self.ligne = ligne_centrale
        self.compteur = 0
        self.next_id = 0
        self.tracks = {}
        
        self.smoother = TemporalSmoother(history_size=5)
        self.iou_threshold = 0.3
        self.max_lost = 8
        self.kalman_enabled = True
        self.min_confidence = 0.3
    
    def _init_kalman(self, bbox):
        """Initialise un filtre de Kalman"""
        kf = cv2.KalmanFilter(6, 2)
        kf.measurementMatrix = np.array([
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0]
        ], np.float32)
        
        kf.transitionMatrix = np.array([
            [1, 0, 0, 0, 1, 0],
            [0, 1, 0, 0, 0, 1],
            [0, 0, 1, 0, 0, 0],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1]
        ], np.float32)
        
        kf.processNoiseCov = np.eye(6, dtype=np.float32) * 0.01
        
        cx = (bbox[0] + bbox[2]) / 2
        cy = (bbox[1] + bbox[3]) / 2
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        
        kf.statePre = np.array([[cx], [cy], [w], [h], [0], [0]], np.float32)
        kf.statePost = np.array([[cx], [cy], [w], [h], [0], [0]], np.float32)
        
        return kf
    
    def _compute_iou(self, box1, box2):
        """Calcule l'IoU entre deux boîtes"""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - inter
        
        return inter / union if union > 0 else 0
    
    def _predict(self):
        """Prédit la position des tracks pour la frame suivante"""
        for track_id in self.tracks:
            if self.kalman_enabled and self.tracks[track_id]['kalman'] is not None:
                kf = self.tracks[track_id]['kalman']
                predicted = kf.predict()
                cx_pred, cy_pred = predicted[0,0], predicted[1,0]
                last_bbox = self.tracks[track_id]['bbox']
                w = last_bbox[2] - last_bbox[0]
                h = last_bbox[3] - last_bbox[1]
                self.tracks[track_id]['predicted_bbox'] = [
                    int(cx_pred - w//2),
                    int(cy_pred - h//2),
                    int(cx_pred + w//2),
                    int(cy_pred + h//2)
                ]
    
    def _associate(self, detections):
        """Associe les détections aux tracks existants"""
        if not self.tracks or not detections:
            return {}, list(range(len(detections))), list(self.tracks.keys())
        
        iou_matrix = np.zeros((len(self.tracks), len(detections)))
        
        for t_idx, track_id in enumerate(self.tracks):
            track_bbox = self.tracks[track_id].get('predicted_bbox', self.tracks[track_id]['bbox'])
            for d_idx, det in enumerate(detections):
                iou_matrix[t_idx, d_idx] = self._compute_iou(track_bbox, det['bbox'])
        
        matched = {}
        unmatched_det = list(range(len(detections)))
        unmatched_track = list(range(len(self.tracks)))
        
        while unmatched_track and unmatched_det:
            max_iou = self.iou_threshold
            best_t = best_d = -1
            
            for t in unmatched_track:
                for d in unmatched_det:
                    if iou_matrix[t, d] > max_iou:
                        max_iou = iou_matrix[t, d]
                        best_t, best_d = t, d
            
            if best_t == -1:
                break
            
            track_id = list(self.tracks.keys())[best_t]
            matched[best_d] = track_id
            unmatched_track.remove(best_t)
            unmatched_det.remove(best_d)
        
        unmatched_track_ids = [list(self.tracks.keys())[i] for i in unmatched_track]
        return matched, unmatched_det, unmatched_track_ids
    
    def _update_kalman(self, track_id, measurement):
        """Met à jour le filtre de Kalman"""
        if not self.kalman_enabled:
            return
        
        kf = self.tracks[track_id]['kalman']
        if kf is None:
            return
        
        kf.predict()
        kf.correct(np.array([[measurement[0]], [measurement[1]]], np.float32))
        
        cx = kf.statePost[0,0]
        cy = kf.statePost[1,0]
        w = kf.statePost[2,0]
        h = kf.statePost[3,0]
        
        self.tracks[track_id]['kalman_bbox'] = [
            int(cx - w/2), int(cy - h/2),
            int(cx + w/2), int(cy + h/2)
        ]
    
    def update(self, detections, frame_shape=None):
        """Met à jour le tracking avec les nouvelles détections"""
        
        if frame_shape is not None:
            self.smoother.update_shape(frame_shape[0], frame_shape[1])
        
        self._predict()
        matched, unmatched_det, unmatched_track = self._associate(detections)
        
        for det_idx, track_id in matched.items():
            det = detections[det_idx]
            smooth_bbox, smooth_conf = self.smoother.smooth(track_id, det['bbox'], det['conf'])
            
            if smooth_conf < self.min_confidence:
                continue
            
            cx = (smooth_bbox[0] + smooth_bbox[2]) // 2
            cy = (smooth_bbox[1] + smooth_bbox[3]) // 2
            
            old_side = self.tracks[track_id]['side']
            new_side = 'below' if cy > self.ligne else 'above'
            
            if old_side != new_side and not self.tracks[track_id]['counted']:
                self.compteur += 1
                self.tracks[track_id]['counted'] = True
                print(f"✅ Personne {track_id} comptée! Total: {self.compteur}")
            
            self.tracks[track_id].update({
                'bbox': smooth_bbox,
                'center': (cx, cy),
                'side': new_side,
                'lost': 0,
                'last_seen': time.time()
            })
            self._update_kalman(track_id, (cx, cy))
        
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
                'kalman': self._init_kalman(det['bbox'])
            }
            self.smoother.smooth(self.next_id, det['bbox'], det['conf'])
            self.next_id += 1
        
        for track_id in unmatched_track:
            self.tracks[track_id]['lost'] += 1
        
        to_delete = [tid for tid, track in self.tracks.items() if track['lost'] > self.max_lost]
        active_tracks = set(self.tracks.keys()) - set(to_delete)
        self.smoother.cleanup(active_tracks)
        
        for tid in to_delete:
            del self.tracks[tid]
        
        current_tracks = []
        for track_id, track in self.tracks.items():
            if track['lost'] == 0:
                bbox = track.get('kalman_bbox', track['bbox'])
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
    
    def preprocess(self, frame):
        """Prétraitement de l'image"""
        h, w = frame.shape[:2]
        self.original_shape = (h, w)
        
        scale = min(self.input_width / w, self.input_height / h)
        new_w = int(w * scale)
        new_h = int(h * scale)
        
        resized = cv2.resize(frame, (new_w, new_h))
        
        pad_x = (self.input_width - new_w) // 2
        pad_y = (self.input_height - new_h) // 2
        
        padded = np.full((self.input_height, self.input_width, 3), 114, dtype=np.uint8)
        padded[pad_y:pad_y+new_h, pad_x:pad_x+new_w] = resized
        
        tensor = np.expand_dims(padded, axis=0)
        tensor = np.ascontiguousarray(tensor)
        
        return tensor, scale, pad_x, pad_y
    
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
            print(f"📊 Frames totales: {total_frames}")
        
        if self.line_y is None:
            self.tracker.ligne = height // 2
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
                    
                    input_tensor, scale, pad_x, pad_y = self.preprocess(frame)
                    
                    try:
                        input_data = {self.input_vstream_info.name: input_tensor}
                        
                        if frame_count == 1:
                            print(f"  Taille du tenseur d'entrée: {input_tensor.nbytes} bytes")
                            print(f"  Shape: {input_tensor.shape}")
                            print(f"  dtype: {input_tensor.dtype}")
                            print(f"  Min/Max: {input_tensor.min()}/{input_tensor.max()}")
                            print(f"  Scale: {scale:.3f}, Pad: ({pad_x}, {pad_y})")
                        
                        output_data = self.infer_pipeline.infer(input_data)
                        
                    except Exception as e:
                        print(f"⚠️ Erreur inférence à la frame {frame_count}: {e}")
                        continue
                    
                    detections = []
                    if output_data and self.output_vstream_info.name in output_data:
                        model_output = output_data[self.output_vstream_info.name]
                        detections = self.postproc.process_yolo_output(
                            model_output,
                            conf_threshold=self.conf_threshold,
                            original_shape=(height, width),
                            input_shape=(self.input_height, self.input_width),
                            scale=scale,
                            pad_x=pad_x,
                            pad_y=pad_y
                        )
                    
                    tracked = self.tracker.update(detections, frame_shape=(height, width))
                    
                    for track_id, det in tracked:
                        bbox = det['bbox']
                        cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
                        cv2.putText(frame, f"ID:{track_id}", (bbox[0], bbox[1]-10),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                    
                    cv2.line(frame, (0, self.tracker.ligne), (frame.shape[1], self.tracker.ligne), (0, 255, 0), 2)
                    
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
    parser.add_argument('--conf', '-c', type=float, default=0.5,
                       help='Seuil de confiance pour les détections (défaut: 0.5)')
    parser.add_argument('--line', '-l', type=int, default=None,
                       help='Position Y de la ligne de comptage (défaut: milieu de l\'image)')
    
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