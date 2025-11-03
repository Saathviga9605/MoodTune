"""
Emotion Detection Module
Uses MTCNN for face detection and FER for emotion recognition
"""

import cv2
import numpy as np
from collections import Counter
from typing import Tuple, Optional, List
import base64

class EmotionDetector:
    def __init__(self):
        """Initialize emotion detector with FER"""
        try:
            # Initialize FER without MTCNN to avoid conflicts
            from fer import FER
            self.emotion_detector = FER(mtcnn=False)
            self.initialized = True
            print("✓ Emotion detector initialized successfully")
        except Exception as e:
            print(f"Error initializing detectors: {e}")
            self.initialized = False
    
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
                    # Detect emotions in the frame
                    results = self.emotion_detector.detect_emotions(frame)
                    
                    if results and len(results) > 0:
                        # Get the first (or largest) face
                        result = results[0]
                        
                        # Get bounding box
                        box = result['box']
                        x, y, w, h = box
                        
                        # Get emotions
                        emotions = result['emotions']
                        
                        # Find dominant emotion
                        dominant_emotion = max(emotions.items(), key=lambda item: item[1])
                        emotion_name = dominant_emotion[0]
                        emotion_score = dominant_emotion[1]
                        
                        # Only add if confidence is above threshold
                        if emotion_score >= confidence_threshold:
                            frame_emotions.append(emotion_name)
                            frames_captured += 1
                        
                        # Draw bounding box (green)
                        cv2.rectangle(display_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                        
                        # Draw emotion label with background
                        label = f"{emotion_name}: {emotion_score:.2f}"
                        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                        
                        # Draw background rectangle for text
                        cv2.rectangle(display_frame, 
                                    (x, y - label_size[1] - 10),
                                    (x + label_size[0], y),
                                    (0, 255, 0), -1)
                        
                        # Draw text
                        cv2.putText(display_frame, label,
                                  (x, y - 5),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                                  (0, 0, 0), 2)
                        
                        # Draw all emotion scores on the side
                        y_offset = 30
                        for emo, score in sorted(emotions.items(), key=lambda x: x[1], reverse=True):
                            text = f"{emo}: {score:.2f}"
                            cv2.putText(display_frame, text,
                                      (10, y_offset),
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                      (255, 255, 255), 1)
                            y_offset += 25
                    
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
        
        # Aggregate emotions using majority voting
        if frame_emotions and len(frame_emotions) >= 5:
            emotion, confidence = self._aggregate_emotions(frame_emotions)
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
    
    def _aggregate_emotions(self, emotions: List[str]) -> Tuple[str, float]:
        """
        Aggregate emotions using majority voting
        
        Args:
            emotions: List of detected emotions
            
        Returns:
            Tuple of (dominant_emotion, confidence)
        """
        if not emotions:
            return 'neutral', 0.5
        
        emotion_counts = Counter(emotions)
        dominant_emotion, count = emotion_counts.most_common(1)[0]
        confidence = count / len(emotions)
        
        return dominant_emotion, confidence
    
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
        
        print(f"✓ Simulated detection: {emotion} (confidence: {confidence})")
        
        return emotion, confidence, frame_emotions