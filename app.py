"""
MoodTune Flask Application - FULLY FIXED & POSSESSIVE
"""

from flask import Flask, render_template, request, jsonify, session, Response
from flask_cors import CORS
import os
from datetime import timedelta
import cv2
import numpy as np
import threading
import time
from collections import deque

from cv.emotion_detector import EmotionDetector
from rl.q_learning import QLearningAgent
from config.api import SpotifyAPI, TMDBAPI
from config.emotions import MINDFULNESS_TIPS, FALLBACK_SONGS, FALLBACK_MOVIES
from utils.helpers import (
    ensure_directories, 
    log_interaction, 
    log_event,
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
emotion_data = {
    'emotion': None,
    'confidence': 0,
    'raw_emotion': None,
    'raw_confidence': 0,
    'stabilized_emotion': None,
    'stabilized_confidence': 0,
    'frame_count': 0,
    'emotions_detected': [],
    'emotion_scores': {},
    'latency_ms': 0.0,
}
detection_active = False
detection_lock = threading.Lock()
emotion_trajectory = deque(maxlen=50)  # Increased from 20 to capture full detection (30 frames)
recommendation_cache = {}
RECOMMENDATION_CACHE_TTL = 300

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
            frame_state = emotion_detector.process_frame(frame)

            if frame_state['face_detected']:
                x, y, w, h = frame_state['face_box']
                raw_emotion = frame_state['raw_emotion']
                raw_confidence = frame_state['raw_confidence']
                stabilized_emotion = frame_state['stabilized_emotion']
                stabilized_confidence = frame_state['stabilized_confidence']

                with detection_lock:
                    emotion_data['raw_emotion'] = raw_emotion
                    emotion_data['raw_confidence'] = raw_confidence
                    emotion_data['stabilized_emotion'] = stabilized_emotion
                    emotion_data['stabilized_confidence'] = stabilized_confidence
                    emotion_data['emotion'] = stabilized_emotion
                    emotion_data['confidence'] = stabilized_confidence
                    emotion_data['emotion_scores'] = frame_state['emotion_scores']
                    emotion_data['latency_ms'] = frame_state['latency_ms']
                    emotion_data['frame_count'] += 1
                    emotion_data['emotions_detected'].append(stabilized_emotion)
                    emotion_trajectory.append({
                        'timestamp': time.time(),
                        'raw_emotion': raw_emotion,
                        'stabilized_emotion': stabilized_emotion,
                        'stabilized_confidence': stabilized_confidence,
                    })

                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 3)

                header_text = f"Raw: {raw_emotion} {raw_confidence:.2f} | Stable: {stabilized_emotion} {stabilized_confidence:.2f}"
                header_size = cv2.getTextSize(header_text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)[0]
                cv2.rectangle(frame, (max(0, x), max(0, y - 44)), (min(frame.shape[1] - 10, x + header_size[0] + 18), y), (0, 255, 0), -1)
                cv2.putText(frame, header_text, (x + 8, y - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1)

                y_offset = 34
                for emo in emotion_detector.emotion_labels:
                    score = frame_state['emotion_scores'].get(emo, 0.0)
                    bar_width = int(score * 160)
                    cv2.rectangle(frame, (10, y_offset - 14), (10 + bar_width, y_offset),
                                  (0, 255, 0) if emo == stabilized_emotion else (90, 90, 90), -1)
                    text = f"{emo}: {score:.2f}"
                    cv2.putText(frame, text, (175, y_offset - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
                    y_offset += 26
            else:
                with detection_lock:
                    emotion_data['latency_ms'] = frame_state['latency_ms']
                cv2.putText(frame, "Position your face in frame", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        except Exception as e:
            print(f"Error in frame processing: {e}")

        with detection_lock:
            progress = min((emotion_data['frame_count'] / required_frames) * 100, 100)
            bar_width = int((frame.shape[1] - 40) * (emotion_data['frame_count'] / required_frames))

        cv2.rectangle(frame, (20, frame.shape[0] - 50), (frame.shape[1] - 20, frame.shape[0] - 30), (50, 50, 50), -1)
        cv2.rectangle(frame, (20, frame.shape[0] - 50), (20 + bar_width, frame.shape[0] - 30), (0, 255, 0), -1)
        progress_text = f"Analyzing: {emotion_data['frame_count']}/{required_frames} ({progress:.0f}%) | Latency {emotion_data.get('latency_ms', 0.0):.0f}ms"
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
    emotion_detector.reset_state()
    emotion_trajectory.clear()
    emotion_data = {
        'emotion': None,
        'confidence': 0,
        'raw_emotion': None,
        'raw_confidence': 0,
        'stabilized_emotion': None,
        'stabilized_confidence': 0,
        'frame_count': 0,
        'emotions_detected': [],
        'emotion_scores': {},
        'latency_ms': 0.0,
    }
    
    # CLEAR SESSION
    session.pop('songs', None)
    session.pop('movies', None)
    session.pop('current_emotion', None)
    session.pop('previous_emotion', None)
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
            dominant_emotion = emotion_data.get('stabilized_emotion') or emotion_data.get('emotion') or 'neutral'
            confidence = float(emotion_data.get('stabilized_confidence') or emotion_data.get('confidence') or 0.0)
            emotion_data['emotion'] = dominant_emotion
            emotion_data['confidence'] = confidence
            detection_active = False
            return jsonify({
                'complete': True,
                'emotion': dominant_emotion,
                'confidence': float(confidence),
                'raw_emotion': emotion_data.get('raw_emotion'),
                'raw_confidence': float(emotion_data.get('raw_confidence', 0.0)),
                'stabilized_emotion': dominant_emotion,
                'stabilized_confidence': float(confidence),
                'emotion_scores': emotion_data.get('emotion_scores', {}),
                'frame_count': emotion_data['frame_count'],
            })
        return jsonify({
            'complete': False,
            'frame_count': emotion_data['frame_count'],
            'progress': min((emotion_data['frame_count'] / 30) * 100, 100),
            'raw_emotion': emotion_data.get('raw_emotion'),
            'raw_confidence': float(emotion_data.get('raw_confidence', 0.0)),
            'stabilized_emotion': emotion_data.get('stabilized_emotion'),
            'stabilized_confidence': float(emotion_data.get('stabilized_confidence', 0.0)),
            'emotion_scores': emotion_data.get('emotion_scores', {}),
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
        # Use the first detected emotion from the trajectory as the starting point for transitions
        # This preserves the emotion sequence from detection (e.g., happy→angry→surprise)
        if emotion_trajectory and not session.get('previous_emotion'):
            first_trajectory_emotion = emotion_trajectory[0].get('stabilized_emotion')
            if first_trajectory_emotion:
                session['previous_emotion'] = first_trajectory_emotion
        session['previous_emotion'] = session.get('previous_emotion') or emotion

        cache_key = emotion.lower()
        cached = recommendation_cache.get(cache_key)
        if cached and (time.time() - cached['timestamp'] < RECOMMENDATION_CACHE_TTL):
            return jsonify(cached['response'])

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

        response_payload = {
            'success': True,
            'emotion': emotion,
            'songs': format_song_data(songs),
            'movies': format_movie_data(movies),
            'mindfulness': mindfulness,
            'rl_stats': rl_agent.get_stats(),
            'analytics': rl_agent.get_analytics(),
        }
        recommendation_cache[cache_key] = {
            'timestamp': time.time(),
            'response': response_payload,
        }
        return jsonify(response_payload)

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
        engagement_score = float(request.json.get('engagement_score', 0.0))
        completion_score = float(request.json.get('completion_score', 1.0))
        mood_improvement_score = float(request.json.get('mood_improvement_score', 0.0))
        if not emotion or not song_id:
            return jsonify({'error': 'Missing required fields'}), 400

        feedback_score = 1.0 if liked else -1.0
        reward = rl_agent.update_q_value(
            emotion,
            song_id,
            content_type='song',
            feedback_score=feedback_score,
            engagement_score=engagement_score,
            mood_improvement_score=mood_improvement_score,
            completion_score=completion_score,
            previous_emotion=session.get('previous_emotion'),
            next_emotion=session.get('current_emotion'),
            metadata={'liked': liked}
        )
        log_interaction(emotion, song_id, reward, metadata={
            'content_type': 'song',
            'liked': liked,
            'engagement_score': engagement_score,
            'completion_score': completion_score,
            'mood_improvement_score': mood_improvement_score,
        })
        log_event('feedback', {
            'emotion': emotion,
            'content_type': 'song',
            'content_id': song_id,
            'reward': reward,
            'engagement_score': engagement_score,
            'completion_score': completion_score,
            'mood_improvement_score': mood_improvement_score,
            'liked': liked,
        })

        songs = session.get('songs', [])
        current_index = session.get('song_index', 0)
        
        # MARK CURRENT AS SHOWN
        rl_agent.mark_shown(emotion, 'song', current_index)

        next_index = rl_agent.select_action(emotion, songs, content_type='song', exclude_indices=[current_index])
        session['song_index'] = next_index
        session['previous_emotion'] = emotion

        return jsonify({
            'success': True,
            'next_song': format_song_data([songs[next_index]])[0],
            'next_index': next_index,
            'rl_stats': rl_agent.get_stats(),
            'analytics': rl_agent.get_analytics(),
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
        engagement_score = float(request.json.get('engagement_score', 0.0))
        completion_score = float(request.json.get('completion_score', 1.0))
        mood_improvement_score = float(request.json.get('mood_improvement_score', 0.0))
        if not emotion or not movie_title:
            return jsonify({'error': 'Missing required fields'}), 400

        feedback_score = 1.0 if liked else -1.0
        reward = rl_agent.update_q_value(
            emotion,
            movie_title,
            content_type='movie',
            feedback_score=feedback_score,
            engagement_score=engagement_score,
            mood_improvement_score=mood_improvement_score,
            completion_score=completion_score,
            previous_emotion=session.get('previous_emotion'),
            next_emotion=session.get('current_emotion'),
            metadata={'liked': liked}
        )
        log_interaction(emotion, movie_title, reward, metadata={
            'content_type': 'movie',
            'liked': liked,
            'engagement_score': engagement_score,
            'completion_score': completion_score,
            'mood_improvement_score': mood_improvement_score,
        })
        log_event('feedback', {
            'emotion': emotion,
            'content_type': 'movie',
            'content_id': movie_title,
            'reward': reward,
            'engagement_score': engagement_score,
            'completion_score': completion_score,
            'mood_improvement_score': mood_improvement_score,
            'liked': liked,
        })

        movies = session.get('movies', [])
        current_index = session.get('movie_index', 0)
        
        # MARK CURRENT AS SHOWN
        rl_agent.mark_shown(emotion, 'movie', current_index)

        next_index = rl_agent.select_action(emotion, movies, content_type='movie', exclude_indices=[current_index])
        session['movie_index'] = next_index
        session['previous_emotion'] = emotion

        return jsonify({
            'success': True,
            'next_movie': format_movie_data([movies[next_index]])[0],
            'next_index': next_index,
            'rl_stats': rl_agent.get_stats(),
            'analytics': rl_agent.get_analytics(),
        })
    except Exception as e:
        print(f"Error in feedback-movie: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/api/stats', methods=['GET'])
def get_stats():
    try:
        return jsonify({'success': True, 'stats': rl_agent.get_stats(), 'analytics': rl_agent.get_analytics()})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/analytics', methods=['GET'])
def get_analytics():
    try:
        return jsonify({'success': True, 'analytics': rl_agent.get_analytics()})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/reset-session', methods=['POST'])
def reset_session():
    global detection_active
    detection_active = False
    session.clear()
    recommendation_cache.clear()
    emotion_trajectory.clear()
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