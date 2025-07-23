#!/usr/bin/env python3
"""
Pose Extraction Service using MediaPipe

Simple and efficient pose extraction using Google's MediaPipe library.
Supports GPU acceleration and provides clean pose keypoint extraction.
"""

import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Union, Tuple, Any
import logging
import cv2
import time
from tqdm import tqdm

# Import common services
from common_fs_service import (
    ResultsManager, 
    validate_config, 
    create_service_runner,
    logger
)

# Configure logging
logging.basicConfig(level=logging.INFO)

# Try to import MediaPipe
try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
    logger.info("MediaPipe library imported successfully")
except ImportError as e:
    logger.warning(f"MediaPipe library not found: {e}")
    logger.warning("Please install MediaPipe: pip install mediapipe")
    MEDIAPIPE_AVAILABLE = False

# Try to import Detectron2
try:
    import detectron2
    from detectron2 import model_zoo
    from detectron2.engine import DefaultPredictor
    from detectron2.config import get_cfg
    from detectron2.utils.visualizer import Visualizer
    from detectron2.data import MetadataCatalog
    import torch
    DETECTRON2_AVAILABLE = True
    logger.info("Detectron2 library imported successfully")
except ImportError as e:
    logger.warning(f"Detectron2 library not found: {e}")
    logger.warning("Please install Detectron2: pip install 'git+https://github.com/facebookresearch/detectron2.git'")
    DETECTRON2_AVAILABLE = False


class PoseExtractionService:
    """
    Simple pose extraction service using MediaPipe.
    
    MediaPipe provides:
    - Easy GPU acceleration
    - Fast and accurate pose detection
    - 33 3D landmarks per person
    - Real-time performance
    """
    
    def __init__(self, results_manager: Optional[ResultsManager] = None):
        """Initialize the Pose Extraction Service."""
        self.results_manager = results_manager
        self.config = None
        self.pose_detector = None
        self.person_detector = None
        self.detectron2_predictor = None
        
    def load_config(self, config_path: Union[str, Path]) -> Dict[str, Any]:
        """Load and validate configuration for pose extraction."""
        config = validate_config(config_path, 'pose_extraction')
        
        # Set defaults
        config.setdefault('pose_method', 'mediapipe')  # 'mediapipe' or 'detectron2'
        config.setdefault('confidence_threshold', 0.5)
        config.setdefault('detection_confidence', 0.5)
        config.setdefault('tracking_confidence', 0.5)
        config.setdefault('use_gpu', True)
        config.setdefault('save_pose_images', True)
        config.setdefault('model_complexity', 1)  # 0=lite, 1=full, 2=heavy (MediaPipe only)
        config.setdefault('sample_percentage', 100)  # Process all images by default
        config.setdefault('detect_multiple_people', True)  # Enable multi-person detection
        config.setdefault('person_detection_confidence', 0.5)  # Confidence for person detection
        # Detectron2 specific settings
        config.setdefault('detectron2_model', 'COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml')
        config.setdefault('detectron2_confidence_threshold', 0.7)
        
        # Convert paths
        config['embeddings_path'] = Path(config['embeddings_path'])
        config['output_path'] = Path(config['output_path'])
        
        self.config = config
        logger.info(f"Configuration loaded from {config_path}")
        return config
    
    def setup_pose_detector(self) -> bool:
        """Setup pose detector based on configured method (MediaPipe or Detectron2)."""
        pose_method = self.config.get('pose_method', 'mediapipe').lower()
        
        if pose_method == 'mediapipe':
            return self._setup_mediapipe()
        elif pose_method == 'detectron2':
            return self._setup_detectron2()
        else:
            raise ValueError(f"Unknown pose method: {pose_method}. Use 'mediapipe' or 'detectron2'")
    
    def _setup_mediapipe(self) -> bool:
        """Setup MediaPipe pose detector."""
        if not MEDIAPIPE_AVAILABLE:
            raise ImportError("MediaPipe not available. Install with: pip install mediapipe")
        
        mp_pose = mp.solutions.pose
        
        # Configure pose detector
        self.pose_detector = mp_pose.Pose(
            static_image_mode=True,  # For image processing
            model_complexity=self.config.get('model_complexity', 1),
            enable_segmentation=False,  # Disable for speed
            min_detection_confidence=self.config.get('detection_confidence', 0.5),
            min_tracking_confidence=self.config.get('tracking_confidence', 0.5)
        )
        
        # Setup person detector for multi-person detection
        detect_multiple = self.config.get('detect_multiple_people', True)
        if detect_multiple:
            try:
                mp_objectron = mp.solutions.objectron
                # Note: MediaPipe doesn't have a direct person detector in all versions
                # We'll use a simple approach with pose detection on image crops
                logger.info("🚀 MediaPipe multi-person detection enabled (using pose detection on image regions)")
            except:
                logger.warning("⚠️  Multi-person detection requested but not fully supported, using single-person mode")
                detect_multiple = False
        
        # Check GPU usage
        use_gpu = self.config.get('use_gpu', True)
        if use_gpu:
            logger.info("🚀 MediaPipe configured with GPU acceleration")
        else:
            logger.info("Using CPU for MediaPipe processing")
            
        return True
    
    def _setup_detectron2(self) -> bool:
        """Setup Detectron2 Keypoint R-CNN detector."""
        if not DETECTRON2_AVAILABLE:
            raise ImportError(
                "Detectron2 not available. Install with: pip install 'git+https://github.com/facebookresearch/detectron2.git'"
            )
        
        # Configure Detectron2
        cfg = get_cfg()
        model_name = self.config.get('detectron2_model', 'COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml')
        cfg.merge_from_file(model_zoo.get_config_file(model_name))
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = self.config.get('detectron2_confidence_threshold', 0.7)
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(model_name)
        
        # GPU/CPU configuration
        use_gpu = self.config.get('use_gpu', True)
        if use_gpu and torch.cuda.is_available():
            cfg.MODEL.DEVICE = "cuda"
            logger.info("🚀 Detectron2 configured with GPU acceleration")
            logger.info(f"   Model: {model_name}")
            logger.info(f"   Confidence threshold: {cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST}")
        else:
            cfg.MODEL.DEVICE = "cpu"
            logger.info("Using CPU for Detectron2 processing")
            if use_gpu and not torch.cuda.is_available():
                logger.warning("GPU requested but CUDA not available")
        
        # Create predictor
        self.detectron2_predictor = DefaultPredictor(cfg)
        
        logger.info("✅ Detectron2 Keypoint R-CNN initialized successfully")
        return True
    
    def load_embeddings(self, embeddings_path: Path) -> Tuple[np.ndarray, List[str]]:
        """Load pre-computed embeddings and optionally sample a subset."""
        if not embeddings_path.exists():
            raise FileNotFoundError(f"Embeddings file not found: {embeddings_path}")
            
        data = np.load(embeddings_path)
        embeddings = data['embeddings']
        image_paths = data['image_paths'].tolist()
        
        total_images = len(image_paths)
        logger.info(f"Loaded {total_images} embeddings from {embeddings_path}")
        
        # Sample subset if specified
        sample_percentage = float(self.config.get('sample_percentage', 100))
        if sample_percentage < 100:
            import random
            
            # Calculate number of samples (handle float percentages like 0.1)
            num_samples = int(total_images * sample_percentage / 100)
            num_samples = max(1, num_samples)  # Ensure at least 1 sample
            
            # Create random indices
            random.seed(42)  # For reproducible sampling
            indices = random.sample(range(total_images), num_samples)
            indices.sort()  # Keep original order
            
            # Sample embeddings and paths
            embeddings = embeddings[indices]
            image_paths = [image_paths[i] for i in indices]
            
            logger.info(f"📊 Sampling {sample_percentage}% of images: {len(image_paths)}/{total_images} images")
        else:
            logger.info(f"📊 Processing all {total_images} images")
        
        return embeddings, image_paths
    
    def extract_pose_from_image(self, image_path: str) -> Dict[str, Any]:
        """Extract pose from a single image using configured method (MediaPipe or Detectron2)."""
        pose_method = self.config.get('pose_method', 'mediapipe').lower()
        
        if pose_method == 'mediapipe':
            return self._extract_pose_mediapipe(image_path)
        elif pose_method == 'detectron2':
            return self._extract_pose_detectron2(image_path)
        else:
            raise ValueError(f"Unknown pose method: {pose_method}")
    
    def _extract_pose_mediapipe(self, image_path: str) -> Dict[str, Any]:
        """Extract pose using MediaPipe, supporting multiple people."""
        try:
            # Load image
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Could not load image: {image_path}")
            
            # Convert BGR to RGB for MediaPipe
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            pose_data = {
                'image_path': image_path,
                'people': [],  # Changed to support multiple people
                'person_count': 0,
                'has_pose': False,
                'pose_method': 'mediapipe'
            }
            
            detect_multiple = self.config.get('detect_multiple_people', True)
            confidence_threshold = self.config.get('confidence_threshold', 0.5)
            
            if detect_multiple:
                # Multi-person detection using grid-based approach
                detected_people = self._detect_multiple_people_mediapipe(image_rgb, confidence_threshold)
            else:
                # Single-person detection (original behavior)
                detected_people = self._detect_single_person_mediapipe(image_rgb, confidence_threshold)
            
            if detected_people:
                pose_data.update({
                    'people': detected_people,
                    'person_count': len(detected_people),
                    'has_pose': True,
                    # For backward compatibility, include first person's data at root level
                    'landmarks': detected_people[0].get('landmarks'),
                    'landmarks_3d': detected_people[0].get('landmarks_3d'),
                    'pose_confidence': detected_people[0].get('pose_confidence', 0.0),
                    'visibility_scores': detected_people[0].get('visibility_scores')
                })
            
            return pose_data
            
        except Exception as e:
            logger.warning(f"Failed to extract pose from {image_path}: {e}")
            return {
                'image_path': image_path,
                'people': [],
                'person_count': 0,
                'has_pose': False,
                'landmarks': None,
                'landmarks_3d': None,
                'pose_confidence': 0.0,
                'visibility_scores': None,
                'pose_method': 'mediapipe',
                'error': str(e)
            }
    
    def _extract_pose_detectron2(self, image_path: str) -> Dict[str, Any]:
        """Extract pose using Detectron2 Keypoint R-CNN."""
        try:
            # Load image
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Could not load image: {image_path}")
            
            # Detectron2 expects BGR format (OpenCV default)
            outputs = self.detectron2_predictor(image)
            
            pose_data = {
                'image_path': image_path,
                'people': [],
                'person_count': 0,
                'has_pose': False,
                'pose_method': 'detectron2'
            }
            
            # Extract instances
            instances = outputs["instances"]
            if len(instances) == 0:
                return pose_data
            
            # Get keypoints, scores, and boxes
            keypoints = instances.pred_keypoints.cpu().numpy()  # Shape: (N, 17, 3)
            scores = instances.scores.cpu().numpy()  # Shape: (N,)
            boxes = instances.pred_boxes.tensor.cpu().numpy()  # Shape: (N, 4)
            
            confidence_threshold = self.config.get('confidence_threshold', 0.5)
            detected_people = []
            
            h, w = image.shape[:2]
            
            for person_idx in range(len(keypoints)):
                person_score = scores[person_idx]
                person_keypoints = keypoints[person_idx]  # Shape: (17, 3)
                person_box = boxes[person_idx]  # [x1, y1, x2, y2]
                
                # Filter by detection confidence
                if person_score < confidence_threshold:
                    continue
                
                # Convert keypoints to our format
                # Detectron2 keypoints: (x, y, visibility) where visibility is 0/1/2
                # Our format: flatten to [x1, y1, x2, y2, ...] with separate visibility scores
                landmarks_2d = []
                visibility_scores = []
                
                for kp in person_keypoints:
                    x, y, vis = kp
                    # Normalize coordinates to [0, 1]
                    landmarks_2d.extend([x / w, y / h])
                    # Convert Detectron2 visibility (0=not visible, 1=occluded, 2=visible) to confidence
                    visibility_scores.append(vis / 2.0)  # Convert to 0-1 range
                
                # Calculate average confidence from keypoint visibilities
                visible_keypoints = [v for v in visibility_scores if v > 0]
                avg_confidence = np.mean(visible_keypoints) if visible_keypoints else 0.0
                
                detected_people.append({
                    'person_id': person_idx,
                    'landmarks': landmarks_2d,
                    'landmarks_3d': None,  # Detectron2 doesn't provide 3D landmarks
                    'pose_confidence': float(max(avg_confidence, person_score)),  # Use max of keypoint confidence and detection score
                    'detection_score': float(person_score),
                    'visibility_scores': visibility_scores,
                    'bounding_box': {
                        'x1': float(person_box[0]),
                        'y1': float(person_box[1]), 
                        'x2': float(person_box[2]),
                        'y2': float(person_box[3])
                    },
                    'keypoint_format': 'coco_17'  # COCO 17-keypoint format
                })
            
            if detected_people:
                # Sort by confidence (highest first)
                detected_people.sort(key=lambda x: x['pose_confidence'], reverse=True)
                
                pose_data.update({
                    'people': detected_people,
                    'person_count': len(detected_people),
                    'has_pose': True,
                    # For backward compatibility, include first person's data at root level
                    'landmarks': detected_people[0].get('landmarks'),
                    'landmarks_3d': None,  # Detectron2 doesn't provide 3D
                    'pose_confidence': detected_people[0].get('pose_confidence', 0.0),
                    'visibility_scores': detected_people[0].get('visibility_scores')
                })
            
            return pose_data
            
        except Exception as e:
            logger.warning(f"Failed to extract pose from {image_path}: {e}")
            return {
                'image_path': image_path,
                'people': [],
                'person_count': 0,
                'has_pose': False,
                'landmarks': None,
                'landmarks_3d': None,
                'pose_confidence': 0.0,
                'visibility_scores': None,
                'pose_method': 'detectron2',
                'error': str(e)
            }
    
    def _detect_single_person_mediapipe(self, image_rgb: np.ndarray, confidence_threshold: float) -> List[Dict[str, Any]]:
        """Detect pose for single person (original behavior)."""
        results = self.pose_detector.process(image_rgb)
        
        if not results.pose_landmarks:
            return []
        
        # Extract landmarks
        landmarks_2d = []
        visibility_scores = []
        
        for landmark in results.pose_landmarks.landmark:
            landmarks_2d.extend([landmark.x, landmark.y])
            visibility_scores.append(landmark.visibility)
        
        # Extract 3D landmarks if available
        landmarks_3d = []
        if results.pose_world_landmarks:
            for landmark in results.pose_world_landmarks.landmark:
                landmarks_3d.extend([landmark.x, landmark.y, landmark.z])
        
        # Calculate average confidence
        avg_confidence = np.mean(visibility_scores)
        
        # Filter by confidence
        if avg_confidence >= confidence_threshold:
            return [{
                'person_id': 0,
                'landmarks': landmarks_2d,
                'landmarks_3d': landmarks_3d if landmarks_3d else None,
                'pose_confidence': float(avg_confidence),
                'visibility_scores': visibility_scores,
                'bounding_box': None  # Full image
            }]
        
        return []
    
    def _detect_multiple_people_mediapipe(self, image_rgb: np.ndarray, confidence_threshold: float) -> List[Dict[str, Any]]:
        """Detect poses for multiple people using grid-based approach."""
        h, w = image_rgb.shape[:2]
        detected_people = []
        
        # First, try full image (most prominent person)
        full_image_people = self._detect_single_person_mediapipe(image_rgb, confidence_threshold)
        if full_image_people:
            detected_people.extend(full_image_people)
        
        # Then try different regions for additional people
        # Grid-based approach: divide image into overlapping regions
        grid_size = 2  # 2x2 grid
        overlap = 0.3  # 30% overlap
        
        for row in range(grid_size):
            for col in range(grid_size):
                # Calculate region boundaries with overlap
                y1 = int(row * h / grid_size * (1 - overlap))
                y2 = int((row + 1) * h / grid_size * (1 + overlap))
                x1 = int(col * w / grid_size * (1 - overlap))
                x2 = int((col + 1) * w / grid_size * (1 + overlap))
                
                # Ensure boundaries are within image
                y1, y2 = max(0, y1), min(h, y2)
                x1, x2 = max(0, x1), min(w, x2)
                
                # Skip if region is too small
                if (y2 - y1) < h/4 or (x2 - x1) < w/4:
                    continue
                
                # Extract region
                region = image_rgb[y1:y2, x1:x2]
                
                # Detect pose in region
                results = self.pose_detector.process(region)
                
                if results.pose_landmarks:
                    # Extract landmarks
                    landmarks_2d = []
                    visibility_scores = []
                    
                    for landmark in results.pose_landmarks.landmark:
                        # Convert relative coordinates back to full image coordinates
                        abs_x = (landmark.x * (x2 - x1) + x1) / w
                        abs_y = (landmark.y * (y2 - y1) + y1) / h
                        landmarks_2d.extend([abs_x, abs_y])
                        visibility_scores.append(landmark.visibility)
                    
                    # Extract 3D landmarks if available
                    landmarks_3d = []
                    if results.pose_world_landmarks:
                        for landmark in results.pose_world_landmarks.landmark:
                            landmarks_3d.extend([landmark.x, landmark.y, landmark.z])
                    
                    # Calculate average confidence
                    avg_confidence = np.mean(visibility_scores)
                    
                    # Filter by confidence and check if it's a new person
                    if avg_confidence >= confidence_threshold:
                        # Check if this person is already detected (avoid duplicates)
                        is_duplicate = self._is_duplicate_person(landmarks_2d, detected_people)
                        
                        if not is_duplicate:
                            detected_people.append({
                                'person_id': len(detected_people),
                                'landmarks': landmarks_2d,
                                'landmarks_3d': landmarks_3d if landmarks_3d else None,
                                'pose_confidence': float(avg_confidence),
                                'visibility_scores': visibility_scores,
                                'bounding_box': {'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2}
                            })
        
        return detected_people
    
    def _is_duplicate_person(self, new_landmarks: List[float], existing_people: List[Dict[str, Any]], 
                           threshold: float = 0.1) -> bool:
        """Check if detected person is a duplicate of an already detected person."""
        if not existing_people:
            return False
        
        new_landmarks_array = np.array(new_landmarks).reshape(-1, 2)
        
        for person in existing_people:
            if person.get('landmarks'):
                existing_landmarks = np.array(person['landmarks']).reshape(-1, 2)
                
                # Calculate average distance between corresponding landmarks
                distances = np.linalg.norm(new_landmarks_array - existing_landmarks, axis=1)
                avg_distance = np.mean(distances)
                
                # If average distance is small, consider it a duplicate
                if avg_distance < threshold:
                    return True
        
        return False
    
    def extract_poses_from_images(self, image_paths: List[str]) -> List[Dict[str, Any]]:
        """Extract poses from all images."""
        all_poses = []
        successful_extractions = 0
        
        logger.info(f"🎯 Processing {len(image_paths)} images with MediaPipe")
        
        with tqdm(image_paths, desc="Extracting poses", unit="img") as pbar:
            total_people_detected = 0
            for i, image_path in enumerate(pbar):
                pose_data = self.extract_pose_from_image(image_path)
                
                if pose_data['has_pose']:
                    successful_extractions += 1
                    total_people_detected += pose_data.get('person_count', 0)
                
                all_poses.append(pose_data)
                
                # Update progress
                success_rate = (successful_extractions / (i + 1)) * 100
                avg_people_per_image = total_people_detected / successful_extractions if successful_extractions > 0 else 0
                pbar.set_postfix({
                    'Success': f"{successful_extractions}/{i + 1}",
                    'Rate': f"{success_rate:.1f}%",
                    'People': f"{total_people_detected}",
                    'Avg': f"{avg_people_per_image:.1f}/img"
                })
        
        logger.info(f"✅ Successfully extracted poses from {successful_extractions}/{len(image_paths)} images")
        return all_poses
    
    def save_pose_visualization(self, image_path: str, pose_data: Dict[str, Any], 
                              output_dir: Path) -> Optional[str]:
        """Save pose visualization for multiple people."""
        if not pose_data['has_pose']:
            return None
            
        try:
            import matplotlib.pyplot as plt
            import matplotlib.colors as mcolors
            
            # Load image
            image = cv2.imread(image_path)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            h, w = image_rgb.shape[:2]
            
            # Create visualization
            fig, ax = plt.subplots(1, 1, figsize=(12, 10))
            ax.imshow(image_rgb)
            
            # Define colors for different people
            colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray']
            
            # Draw poses for all detected people
            people = pose_data.get('people', [])
            for person_idx, person in enumerate(people):
                if not person.get('landmarks'):
                    continue
                    
                # Get landmarks and visibility
                landmarks = np.array(person['landmarks']).reshape(-1, 2)
                visibility = person.get('visibility_scores', [])
                color = colors[person_idx % len(colors)]
                
                # Draw landmarks
                for i, (x, y) in enumerate(landmarks):
                    if i < len(visibility) and visibility[i] > 0.5:  # Only draw visible landmarks
                        ax.plot(x * w, y * h, 'o', color=color, markersize=4)
                
                # Draw connections (simplified skeleton)
                connections = [
                    # Body
                    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),  # Arms
                    (11, 23), (12, 24), (23, 24),  # Torso
                    (23, 25), (25, 27), (24, 26), (26, 28),  # Legs
                    # Face
                    (0, 1), (1, 2), (2, 3), (3, 7),
                    (0, 4), (4, 5), (5, 6), (6, 8)
                ]
                
                for start_idx, end_idx in connections:
                    if (start_idx < len(landmarks) and end_idx < len(landmarks) and
                        start_idx < len(visibility) and end_idx < len(visibility) and
                        visibility[start_idx] > 0.5 and visibility[end_idx] > 0.5):
                        x1, y1 = landmarks[start_idx] * [w, h]
                        x2, y2 = landmarks[end_idx] * [w, h]
                        ax.plot([x1, x2], [y1, y2], '-', color=color, linewidth=2)
                
                # Add person label
                if landmarks.size > 0:
                    # Use shoulder/head position for label
                    head_landmarks = landmarks[:5]  # First 5 are head/face landmarks
                    if len(head_landmarks) > 0:
                        label_x = np.mean(head_landmarks[:, 0]) * w
                        label_y = (np.min(head_landmarks[:, 1]) - 0.05) * h
                        confidence = person.get('pose_confidence', 0.0)
                        ax.text(label_x, label_y, f"Person {person_idx+1}\n{confidence:.2f}", 
                               color=color, fontsize=10, ha='center', weight='bold',
                               bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
            
            # Set title
            image_name = Path(image_path).name
            person_count = pose_data.get('person_count', 0)
            avg_confidence = np.mean([p.get('pose_confidence', 0) for p in people]) if people else 0
            ax.set_title(f"{image_name}\nPeople: {person_count}, Avg Confidence: {avg_confidence:.3f}")
            ax.axis('off')
            
            # Save
            vis_filename = f"pose_{Path(image_path).stem}.png"
            vis_path = output_dir / vis_filename
            plt.savefig(vis_path, bbox_inches='tight', dpi=150)
            plt.close()
            
            return str(vis_path)
            
        except Exception as e:
            logger.warning(f"Failed to save visualization for {image_path}: {e}")
            return None
    
    def save_results(self, all_poses: List[Dict[str, Any]], config: Dict[str, Any]) -> Dict[str, Any]:
        """Save pose extraction results."""
        results_dir = self.results_manager.get_results_dir()
        
        # Save pose data
        poses_path = results_dir / "pose_data.json"
        with open(poses_path, 'w') as f:
            json.dump(all_poses, f, indent=2, default=str)
        
        # Save landmarks as numpy array
        poses_with_landmarks = [p for p in all_poses if p['has_pose']]
        if poses_with_landmarks:
            landmarks_list = []
            landmarks_3d_list = []
            image_paths_with_poses = []
            
            for pose_data in poses_with_landmarks:
                landmarks_list.append(pose_data['landmarks'])
                if pose_data['landmarks_3d']:
                    landmarks_3d_list.append(pose_data['landmarks_3d'])
                image_paths_with_poses.append(pose_data['image_path'])
            
            # Save 2D landmarks
            landmarks_path = results_dir / "pose_landmarks.npz"
            save_data = {
                'landmarks_2d': np.array(landmarks_list),
                'image_paths': image_paths_with_poses
            }
            
            # Add 3D landmarks if available
            if landmarks_3d_list:
                save_data['landmarks_3d'] = np.array(landmarks_3d_list)
            
            np.savez_compressed(landmarks_path, **save_data)
        
        # Generate statistics
        total_images = len(all_poses)
        successful_poses = len(poses_with_landmarks)
        total_people = sum(p.get('person_count', 0) for p in all_poses)
        avg_confidence = np.mean([p['pose_confidence'] for p in poses_with_landmarks]) if poses_with_landmarks else 0.0
        avg_people_per_image = total_people / successful_poses if successful_poses > 0 else 0.0
        
        pose_method = config.get('pose_method', 'mediapipe')
        stats = {
            'total_images_processed': total_images,
            'successful_pose_extractions': successful_poses,
            'success_rate': successful_poses / total_images if total_images > 0 else 0.0,
            'average_pose_confidence': float(avg_confidence),
            'total_people_detected': total_people,
            'average_people_per_image': float(avg_people_per_image),
            'pose_extraction_method': pose_method,
            'confidence_threshold': config.get('confidence_threshold', 0.5),
            'detect_multiple_people': config.get('detect_multiple_people', True),
            'person_detection_confidence': config.get('person_detection_confidence', 0.5)
        }
        
        # Add method-specific settings
        if pose_method == 'mediapipe':
            stats['model_complexity'] = config.get('model_complexity', 1)
        elif pose_method == 'detectron2':
            stats['detectron2_model'] = config.get('detectron2_model', 'COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml')
            stats['detectron2_confidence_threshold'] = config.get('detectron2_confidence_threshold', 0.7)
        
        stats_path = results_dir / "pose_statistics.json"
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)
        
        # Save visualizations
        visualization_paths = []
        if config.get('save_pose_images', False) and poses_with_landmarks:
            vis_dir = results_dir / "pose_visualizations"
            vis_dir.mkdir(exist_ok=True)
            
            # Limit to first 20 for performance
            poses_to_visualize = poses_with_landmarks[:20]
            logger.info(f"Saving {len(poses_to_visualize)} pose visualizations...")
            
            for pose_data in tqdm(poses_to_visualize, desc="Saving visualizations"):
                vis_path = self.save_pose_visualization(
                    pose_data['image_path'], pose_data, vis_dir
                )
                if vis_path:
                    visualization_paths.append(vis_path)
        
        # Save configuration and metadata
        additional_metadata = {
            'total_images_processed': total_images,
            'successful_pose_extractions': successful_poses,
            'success_rate': stats['success_rate'],
            'average_pose_confidence': stats['average_pose_confidence'],
            'pose_extraction_method': pose_method,
            'visualizations_created': len(visualization_paths),
            'method': 'pose_extraction'
        }
        
        config_path = self.results_manager.save_run_config(config, additional_metadata)
        
        result_metadata = self.results_manager.save_result_metadata({
            'total_images_processed': total_images,
            'successful_pose_extractions': successful_poses,
            'success_rate': stats['success_rate'],
            'files_created': [
                str(poses_path),
                str(stats_path),
                str(config_path)
            ] + visualization_paths
        })
        
        logger.info(f"Results saved to: {results_dir}")
        logger.info(f"Extracted poses from {successful_poses}/{total_images} images "
                   f"(success rate: {stats['success_rate']:.2%})")
        logger.info(f"Total people detected: {total_people} "
                   f"(avg: {avg_people_per_image:.1f} people/image)")
        
        return result_metadata
    
    def extract_poses(self, config_path: Union[str, Path], run_id: Optional[str] = None) -> str:
        """Main method to extract poses."""
        # Load configuration
        config = self.load_config(config_path)
        
        # Create results manager
        if self.results_manager is None:
            self.results_manager = create_service_runner('pose_extraction', config['output_path'])
        
        actual_run_id = self.results_manager.create_run(run_id)
        
        # Setup pose detector
        self.setup_pose_detector()
        
        # Load embeddings and image paths
        embeddings, image_paths = self.load_embeddings(config['embeddings_path'])
        
        # Process images
        start_time = time.time()
        all_poses = self.extract_poses_from_images(image_paths)
        processing_time = time.time() - start_time
        
        # Performance summary
        successful = sum(1 for p in all_poses if p['has_pose'])
        total_people = sum(p.get('person_count', 0) for p in all_poses)
        images_per_second = len(image_paths) / processing_time if processing_time > 0 else 0
        avg_people_per_image = total_people / successful if successful > 0 else 0
        
        logger.info(f"📊 Performance Summary:")
        logger.info(f"   Total time: {processing_time:.1f}s")
        logger.info(f"   Speed: {images_per_second:.1f} images/second")
        logger.info(f"   Success rate: {successful}/{len(image_paths)} ({successful/len(image_paths)*100:.1f}%)")
        logger.info(f"   People detected: {total_people} (avg: {avg_people_per_image:.1f}/image)")
        
        # Save results
        self.save_results(all_poses, config)
        
        # Cleanup
        if self.pose_detector:
            self.pose_detector.close()
        if self.person_detector:
            self.person_detector.close()
        # Detectron2 doesn't need explicit cleanup
        
        return str(self.results_manager.get_results_dir())


def extract_poses(config_path: Union[str, Path], run_id: Optional[str] = None) -> str:
    """Convenience function to extract poses."""
    service = PoseExtractionService()
    return service.extract_poses(config_path, run_id)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Extract human poses using MediaPipe")
    parser.add_argument("config", help="Path to YAML configuration file")
    parser.add_argument("--run-id", help="Optional run identifier")
    
    args = parser.parse_args()
    
    try:
        service = PoseExtractionService()
        results_path = service.extract_poses(args.config, args.run_id)
        print(f"✅ Pose extraction completed successfully!")
        print(f"   Results saved to: {results_path}")
    except Exception as e:
        print(f"❌ Error: {e}")
        exit(1)
