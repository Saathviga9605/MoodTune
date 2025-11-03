"""
MoodTune Flask Application - FULLY FIXED & POSSESSIVE
"""

from flask import Flask, render_template, request, jsonify, session, Response
from flask_cors import CORS
import os
from datetime import timedelta
import cv2
import base64
import numpy as np
import threading
import time

from cv.emotion_detector import EmotionDetector
from rl.q_learning import QLearningAgent
from config.api import SpotifyAPI, TMDBAPI
from config.emotions import MINDFULNESS_TIPS, FALLBACK_SONGS, FALLBACK_MOVIES
from utils.helpers import (
    ensure_directories, 
    log_interaction, 
    format_song_data, 
    format_movie_data,
    validate_emotion
)

# Initialize Flask app
app = Flask(__name__)
app.secret_key = os.getenv('SECRET_KEY', 'moodtune_secret_key_2024')
app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(hours=24)
CORS(app)

# Initialize components
ensure_directories()
emotion_detector = EmotionDetector()
rl_agent = QLearningAgent()
spotify_api = SpotifyAPI()
tmdb_api = TMDBAPI()

# Global variables for video streaming
camera = None
emotion_data = {'emotion': None, 'confidence': 0, 'frame_count': 0, 'emotions_detected': []}
detection_active = False
detection_lock = threading.Lock()

@app.route('/')
def index():
    return render_template('index.html')

def generate_frames():
    global camera, emotion_data, detection_active
    camera = cv2.VideoCapture(0)
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    required_frames = 30

    while detection_active:
        success, frame = camera.read()
        if not success:
            break
        frame = cv2.flip(frame, 1)

        try:
            results = emotion_detector.emotion_detector.detect_emotions(frame)
            if results and len(results) > 0:
                result = results[0]
                box = result['box']
                x, y, w, h = box
                emotions = result['emotions']
                dominant_emotion = max(emotions.items(), key=lambda item: item[1])
                emotion_name = dominant_emotion[0]
                emotion_score = dominant_emotion[1]

                if emotion_score >= 0.3:
                    with detection_lock:
                        emotion_data['emotions_detected'].append(emotion_name)
                        emotion_data['frame_count'] = len(emotion_data['emotions_detected'])

                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 3)
                label = f"{emotion_name}: {emotion_score:.2f}"
                label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
                cv2.rectangle(frame, (x, y - label_size[1] - 15), (x + label_size[0] + 10, y), (0, 255, 0), -1)
                cv2.putText(frame, label, (x + 5, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

                y_offset = 40
                for emo, score in sorted(emotions.items(), key=lambda x: x[1], reverse=True):
                    bar_width = int(score * 150)
                    cv2.rectangle(frame, (10, y_offset - 15), (10 + bar_width, y_offset), 
                                (0, 255, 0) if emo == emotion_name else (100, 100, 100), -1)
                    text = f"{emo}: {score:.2f}"
                    cv2.putText(frame, text, (165, y_offset - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    y_offset += 30
            else:
                cv2.putText(frame, "Position your face in frame", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        except Exception as e:
            print(f"Error in frame processing: {e}")

        with detection_lock:
            progress = min((emotion_data['frame_count'] / required_frames) * 100, 100)
            bar_width = int((frame.shape[1] - 40) * (emotion_data['frame_count'] / required_frames))

        cv2.rectangle(frame, (20, frame.shape[0] - 50), (frame.shape[1] - 20, frame.shape[0] - 30), (50, 50, 50), -1)
        cv2.rectangle(frame, (20, frame.shape[0] - 50), (20 + bar_width, frame.shape[0] - 30), (0, 255, 0), -1)
        progress_text = f"Analyzing: {emotion_data['frame_count']}/{required_frames} ({progress:.0f}%)"
        cv2.putText(frame, progress_text, (25, frame.shape[0] - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

        if emotion_data['frame_count'] >= required_frames:
            time.sleep(0.5)
            break

    if camera:
        camera.release()

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# In /api/start-detection — ADD RESET
@app.route('/api/start-detection', methods=['POST'])
def start_detection():
    global detection_active, emotion_data
    detection_active = True
    emotion_data = {'emotion': None, 'confidence': 0, 'frame_count': 0, 'emotions_detected': []}
    
    # CLEAR SESSION
    session.pop('songs', None)
    session.pop('movies', None)
    session.pop('current_emotion', None)
    session.pop('song_index', None)
    session.pop('movie_index', None)
    
    # CLEAR SHOWN HISTORY FOR FRESH START
    rl_agent.reset_shown('', 'song')
    rl_agent.reset_shown('', 'movie')
    
    return jsonify({'success': True, 'message': 'Detection started'})

@app.route('/api/get-detection-status', methods=['GET'])
def get_detection_status():
    global emotion_data, detection_active
    with detection_lock:
        if emotion_data['frame_count'] >= 30 and emotion_data['emotions_detected']:
            from collections import Counter
            emotion_counts = Counter(emotion_data['emotions_detected'])
            dominant_emotion, count = emotion_counts.most_common(1)[0]
            confidence = count / len(emotion_data['emotions_detected'])
            emotion_data['emotion'] = dominant_emotion
            emotion_data['confidence'] = confidence
            detection_active = False
            return jsonify({
                'complete': True,
                'emotion': dominant_emotion,
                'confidence': float(confidence),
                'frame_count': emotion_data['frame_count']
            })
        return jsonify({
            'complete': False,
            'frame_count': emotion_data['frame_count'],
            'progress': min((emotion_data['frame_count'] / 30) * 100, 100)
        })

@app.route('/api/stop-detection', methods=['POST'])
def stop_detection():
    global detection_active
    detection_active = False
    if camera:
        camera.release()
    return jsonify({'success': True})

@app.route('/api/get-recommendations', methods=['POST'])
def get_recommendations():
    try:
        emotion = request.json.get('emotion')
        if not emotion or not validate_emotion(emotion):
            return jsonify({'error': 'Invalid emotion'}), 400

        session['current_emotion'] = emotion

        # Fetch songs
        songs = spotify_api.search_songs_by_emotion(emotion, limit=20)
        if not songs:
            songs = FALLBACK_SONGS.get(emotion, [])[:20]

        # Fetch movies
        movies = tmdb_api.search_movies_by_emotion(emotion, limit=15)
        if not movies:
            movies = FALLBACK_MOVIES.get(emotion, [])[:15]

        # Get mindfulness
        mindfulness = MINDFULNESS_TIPS.get(emotion, ["Take a deep breath.", "You're doing great."])

        # Store in session
        session['songs'] = songs
        session['movies'] = movies
        session['song_index'] = 0
        session['movie_index'] = 0

        return jsonify({
            'success': True,
            'emotion': emotion,
            'songs': format_song_data(songs),
            'movies': format_movie_data(movies),
            'mindfulness': mindfulness,
            'rl_stats': rl_agent.get_stats()
        })

    except Exception as e:
        print(f"Error in get_recommendations: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/api/next-song', methods=['POST'])
def next_song():
    try:
        emotion = session.get('current_emotion')
        songs = session.get('songs', [])
        current_index = session.get('song_index', 0)
        if not emotion or not songs:
            return jsonify({'error': 'No active session'}), 400

        next_index = rl_agent.select_action(emotion, songs, content_type='song', exclude_indices=[current_index])
        session['song_index'] = next_index

        return jsonify({
            'success': True,
            'song': format_song_data([songs[next_index]])[0],
            'index': next_index,
            'rl_stats': rl_agent.get_stats()
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/next-movie', methods=['POST'])
def next_movie():
    try:
        emotion = session.get('current_emotion')
        movies = session.get('movies', [])
        current_index = session.get('movie_index', 0)
        if not emotion or not movies:
            return jsonify({'error': 'No active session'}), 400

        next_index = rl_agent.select_action(emotion, movies, content_type='movie', exclude_indices=[current_index])
        session['movie_index'] = next_index

        return jsonify({
            'success': True,
            'movie': format_movie_data([movies[next_index]])[0],
            'index': next_index,
            'rl_stats': rl_agent.get_stats()
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# In /api/feedback-song — MARK CURRENT AS SHOWN
@app.route('/api/feedback-song', methods=['POST'])
def feedback_song():
    try:
        emotion = request.json.get('emotion')
        song_id = request.json.get('song_id')
        liked = request.json.get('liked', False)
        if not emotion or not song_id:
            return jsonify({'error': 'Missing required fields'}), 400

        reward = 1.0 if liked else -0.5
        rl_agent.update_q_value(emotion, song_id, reward, content_type='song')
        log_interaction(emotion, song_id, reward)

        songs = session.get('songs', [])
        current_index = session.get('song_index', 0)
        
        # MARK CURRENT AS SHOWN
        rl_agent.mark_shown(emotion, 'song', current_index)

        next_index = rl_agent.select_action(emotion, songs, content_type='song', exclude_indices=[current_index])
        session['song_index'] = next_index

        return jsonify({
            'success': True,
            'next_song': format_song_data([songs[next_index]])[0],
            'next_index': next_index,
            'rl_stats': rl_agent.get_stats()
        })
    except Exception as e:
        print(f"Error in feedback-song: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

# In /api/feedback-movie — MARK CURRENT AS SHOWN
@app.route('/api/feedback-movie', methods=['POST'])
def feedback_movie():
    try:
        emotion = request.json.get('emotion')
        movie_title = request.json.get('movie_title')
        liked = request.json.get('liked', False)
        if not emotion or not movie_title:
            return jsonify({'error': 'Missing required fields'}), 400

        reward = 1.0 if liked else -0.5
        rl_agent.update_q_value(emotion, movie_title, reward, content_type='movie')
        log_interaction(emotion, movie_title, reward)

        movies = session.get('movies', [])
        current_index = session.get('movie_index', 0)
        
        # MARK CURRENT AS SHOWN
        rl_agent.mark_shown(emotion, 'movie', current_index)

        next_index = rl_agent.select_action(emotion, movies, content_type='movie', exclude_indices=[current_index])
        session['movie_index'] = next_index

        return jsonify({
            'success': True,
            'next_movie': format_movie_data([movies[next_index]])[0],
            'next_index': next_index,
            'rl_stats': rl_agent.get_stats()
        })
    except Exception as e:
        print(f"Error in feedback-movie: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/api/stats', methods=['GET'])
def get_stats():
    try:
        return jsonify({'success': True, 'stats': rl_agent.get_stats()})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/reset-session', methods=['POST'])
def reset_session():
    global detection_active
    detection_active = False
    session.clear()
    return jsonify({'success': True, 'message': 'Session reset'})

@app.errorhandler(404)
def not_found(e):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def server_error(e):
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    print("=" * 60)
    print("MoodTune - AI-Powered Emotion-Based Recommendations")
    print("=" * 60)
    print("\nStarting server...")
    print("Open http://localhost:5000 in your browser")
    print("Press Ctrl+C to stop\n")
    app.run(host='0.0.0.0', port=5000, debug=True, threaded=True)