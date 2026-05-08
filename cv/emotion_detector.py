"""
Emotion Detection Module
Uses FER for real-time facial emotion recognition with temporal stabilization.
"""

import cv2
import numpy as np
import base64
import time
from collections import deque
from typing import Tuple, Optional, List, Dict

class EmotionDetector:
    def __init__(self):
        """Initialize emotion detector with FER"""
        try:
            # Initialize FER without MTCNN to avoid conflicts
            from fer import FER
            self.emotion_detector = FER(mtcnn=False)
            self.emotion_labels = ['happy', 'sad', 'angry', 'neutral', 'surprise', 'fear', 'disgust']
            self.alpha = 0.72
            self.decay_factor = 0.94
            self.hysteresis_threshold = 0.15
            self.confidence_threshold = 0.3
            self.emotion_scores = {emotion: 0.0 for emotion in self.emotion_labels}
            self.stabilized_emotion = 'neutral'
            self.stabilized_confidence = 0.0
            self.raw_emotion = 'neutral'
            self.raw_confidence = 0.0
            self.emotion_history = deque(maxlen=20)
            self.initialized = True
            print("✓ Emotion detector initialized successfully")
        except Exception as e:
            print(f"Error initializing detectors: {e}")
            self.initialized = False

    def reset_state(self):
        """Reset rolling emotion state between sessions."""
        self.emotion_scores = {emotion: 0.0 for emotion in self.emotion_labels}
        self.stabilized_emotion = 'neutral'
        self.stabilized_confidence = 0.0
        self.raw_emotion = 'neutral'
        self.raw_confidence = 0.0
        self.emotion_history.clear()

    def _build_score_vector(self, emotions: Dict[str, float]) -> Dict[str, float]:
        return {emotion: float(emotions.get(emotion, 0.0)) for emotion in self.emotion_labels}

    def _update_rolling_scores(self, current_scores: Dict[str, float]) -> None:
        for emotion in self.emotion_labels:
            previous_score = self.emotion_scores.get(emotion, 0.0)
            current_score = current_scores.get(emotion, 0.0)
            updated_score = self.alpha * current_score + (1.0 - self.alpha) * previous_score
            if current_score <= 0.0:
                updated_score *= self.decay_factor
            self.emotion_scores[emotion] = max(0.0, min(1.0, updated_score))

    def _apply_hysteresis(self, candidate_emotion: str) -> str:
        candidate_score = self.emotion_scores.get(candidate_emotion, 0.0)
        current_score = self.emotion_scores.get(self.stabilized_emotion, 0.0)

        if self.stabilized_emotion not in self.emotion_scores:
            self.stabilized_emotion = candidate_emotion
            return candidate_emotion

        if candidate_emotion != self.stabilized_emotion and candidate_score > current_score + self.hysteresis_threshold:
            self.stabilized_emotion = candidate_emotion

        self.stabilized_confidence = self.emotion_scores.get(self.stabilized_emotion, 0.0)
        return self.stabilized_emotion

    def process_frame(self, frame, confidence_threshold: float = 0.3) -> Dict:
        """Process one camera frame and return raw + stabilized affect state."""
        start_time = time.time()
        result = {
            'face_detected': False,
            'face_box': None,
            'raw_emotion': self.raw_emotion,
            'raw_confidence': self.raw_confidence,
            'stabilized_emotion': self.stabilized_emotion,
            'stabilized_confidence': self.stabilized_confidence,
            'emotion_scores': self.emotion_scores.copy(),
            'latency_ms': 0.0,
        }

        if not self.initialized:
            result['latency_ms'] = (time.time() - start_time) * 1000.0
            return result

        try:
            detections = self.emotion_detector.detect_emotions(frame)
            if not detections:
                for emotion in self.emotion_labels:
                    self.emotion_scores[emotion] *= self.decay_factor
                self.stabilized_confidence = self.emotion_scores.get(self.stabilized_emotion, 0.0)
                result['emotion_scores'] = self.emotion_scores.copy()
                result['latency_ms'] = (time.time() - start_time) * 1000.0
                self.emotion_history.append({
                    'timestamp': time.time(),
                    'raw_emotion': self.raw_emotion,
                    'stabilized_emotion': self.stabilized_emotion,
                    'emotion_scores': self.emotion_scores.copy(),
                })
                return result

            detection = detections[0]
            emotions = self._build_score_vector(detection.get('emotions', {}))
            dominant_emotion, dominant_confidence = max(emotions.items(), key=lambda item: item[1])

            self.raw_emotion = dominant_emotion
            self.raw_confidence = float(dominant_confidence)
            self._update_rolling_scores(emotions)
            self._apply_hysteresis(max(self.emotion_scores.items(), key=lambda item: item[1])[0])

            if self.raw_confidence < confidence_threshold:
                for emotion in self.emotion_labels:
                    self.emotion_scores[emotion] *= self.decay_factor
                self.stabilized_confidence = self.emotion_scores.get(self.stabilized_emotion, 0.0)

            result.update({
                'face_detected': True,
                'face_box': detection.get('box'),
                'raw_emotion': self.raw_emotion,
                'raw_confidence': self.raw_confidence,
                'stabilized_emotion': self.stabilized_emotion,
                'stabilized_confidence': self.stabilized_confidence,
                'emotion_scores': self.emotion_scores.copy(),
            })
            self.emotion_history.append({
                'timestamp': time.time(),
                'raw_emotion': self.raw_emotion,
                'stabilized_emotion': self.stabilized_emotion,
                'emotion_scores': self.emotion_scores.copy(),
            })
        except Exception as e:
            print(f"Error processing emotion frame: {e}")

        result['latency_ms'] = (time.time() - start_time) * 1000.0
        return result
    
    def detect_emotion_from_camera(self, 
                                   num_frames: int = 30,
                                   confidence_threshold: float = 0.3) -> Tuple[str, float, List]:
        """
        Detect emotion from webcam feed with live display
        
        Args:
            num_frames: Number of frames to analyze
            confidence_threshold: Minimum confidence for detection
            
        Returns:
            Tuple of (emotion, confidence, frame_emotions)
        """
        if not self.initialized:
            print("Detector not initialized, using simulation")
            return self._simulate_detection()
        
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("Error: Cannot open camera")
            return self._simulate_detection()
        
        # Set camera properties for better performance
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        frame_emotions = []
        frames_captured = 0
        window_name = 'MoodTune - Emotion Detection'
        self.reset_state()
        
        try:
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            
            print(f"Starting emotion detection... Analyzing {num_frames} frames")
            
            while frames_captured < num_frames:
                ret, frame = cap.read()
                
                if not ret:
                    print("Error: Cannot read frame")
                    break
                
                # Flip frame horizontally for mirror effect
                frame = cv2.flip(frame, 1)
                
                # Create a copy for display
                display_frame = frame.copy()
                
                try:
                    frame_state = self.process_frame(frame, confidence_threshold)

                    if frame_state['face_detected']:
                        box = frame_state['face_box']
                        x, y, w, h = box
                        emotion_name = frame_state['raw_emotion']
                        emotion_score = frame_state['raw_confidence']

                        if emotion_score >= confidence_threshold:
                            frame_emotions.append(frame_state['stabilized_emotion'])
                            frames_captured += 1

                        cv2.rectangle(display_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

                        raw_label = f"Raw: {emotion_name} {emotion_score:.2f}"
                        stable_label = f"Stable: {frame_state['stabilized_emotion']} {frame_state['stabilized_confidence']:.2f}"
                        raw_size = cv2.getTextSize(raw_label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
                        stable_size = cv2.getTextSize(stable_label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
                        label_width = max(raw_size[0], stable_size[0])
                        cv2.rectangle(display_frame,
                                      (x, max(0, y - 42)),
                                      (x + label_width + 14, y),
                                      (0, 255, 0), -1)
                        cv2.putText(display_frame, raw_label, (x + 6, y - 24), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
                        cv2.putText(display_frame, stable_label, (x + 6, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

                        y_offset = 30
                        for emo in self.emotion_labels:
                            score = frame_state['emotion_scores'].get(emo, 0.0)
                            text = f"{emo}: {score:.2f}"
                            cv2.putText(display_frame, text,
                                      (10, y_offset),
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                      (255, 255, 255), 1)
                            y_offset += 22

                    else:
                        # No face detected
                        cv2.putText(display_frame, "No face detected",
                                  (10, 30),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                                  (0, 0, 255), 2)
                
                except Exception as e:
                    print(f"Error processing frame: {e}")
                
                # Draw progress bar
                progress = (frames_captured / num_frames) * 100
                bar_width = int((display_frame.shape[1] - 40) * (frames_captured / num_frames))
                cv2.rectangle(display_frame, (20, display_frame.shape[0] - 40),
                            (20 + bar_width, display_frame.shape[0] - 20),
                            (0, 255, 0), -1)
                cv2.rectangle(display_frame, (20, display_frame.shape[0] - 40),
                            (display_frame.shape[1] - 20, display_frame.shape[0] - 20),
                            (255, 255, 255), 2)
                
                # Draw progress text
                progress_text = f"Progress: {frames_captured}/{num_frames} ({progress:.0f}%)"
                cv2.putText(display_frame, progress_text,
                          (20, display_frame.shape[0] - 45),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                          (255, 255, 255), 2)
                
                # Show the frame
                cv2.imshow(window_name, display_frame)
                
                # Break on 'q' or 'ESC'
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == 27:
                    print("Detection cancelled by user")
                    break
        
        finally:
            cap.release()
            cv2.destroyAllWindows()
            cv2.waitKey(1)  # Extra waitKey to ensure windows close
        
        # Aggregate emotions using stabilized scores
        if frame_emotions and len(frame_emotions) >= 5:
            emotion = self.stabilized_emotion
            confidence = float(self.stabilized_confidence)
            print(f"✓ Detected emotion: {emotion} (confidence: {confidence:.2f})")
            return emotion, confidence, frame_emotions
        else:
            print("Not enough valid frames detected, using simulation")
            return self._simulate_detection()
    
    def detect_emotion_from_image(self, image_path: str) -> Tuple[str, float]:
        """
        Detect emotion from a static image
        
        Args:
            image_path: Path to image file
            
        Returns:
            Tuple of (emotion, confidence)
        """
        if not self.initialized:
            return self._simulate_detection()[0:2]
        
        try:
            image = cv2.imread(image_path)
            
            if image is None:
                print(f"Error: Cannot load image from {image_path}")
                return self._simulate_detection()[0:2]
            
            # Detect emotions
            results = self.emotion_detector.detect_emotions(image)
            
            if results and len(results) > 0:
                result = results[0]
                emotions = result['emotions']
                dominant_emotion = max(emotions.items(), key=lambda item: item[1])
                
                return dominant_emotion[0], dominant_emotion[1]
            else:
                print("No face detected in image")
                return self._simulate_detection()[0:2]
        
        except Exception as e:
            print(f"Error detecting emotion from image: {e}")
            return self._simulate_detection()[0:2]
    
    def _simulate_detection(self) -> Tuple[str, float, List]:
        """
        Simulate emotion detection for demo purposes
        Used when camera is not available
        """
        import random
        
        emotions = ['happy', 'sad', 'angry', 'neutral', 'surprise', 'fear', 'disgust']
        emotion = random.choice(emotions)
        confidence = round(random.uniform(0.75, 0.95), 2)
        
        # Simulate frame emotions
        frame_emotions = [emotion] * 20 + [random.choice(emotions) for _ in range(10)]
        
        self.reset_state()
        self.raw_emotion = emotion
        self.raw_confidence = confidence
        self.stabilized_emotion = emotion
        self.stabilized_confidence = confidence
        self.emotion_scores[emotion] = confidence
        self.emotion_history.append({
            'timestamp': time.time(),
            'raw_emotion': emotion,
            'stabilized_emotion': emotion,
            'emotion_scores': self.emotion_scores.copy(),
        })

        print(f"✓ Simulated detection: {emotion} (confidence: {confidence})")
        
        return emotion, confidence, frame_emotions 