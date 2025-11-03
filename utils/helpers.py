"""
Utility Helper Functions
"""

import os
import json
from datetime import datetime
from typing import Dict, List

def ensure_directories():
    """Ensure required directories exist"""
    directories = ['data', 'logs', 'static/uploads']
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)

def log_interaction(emotion: str, song_id: str, reward: float):
    """Log user interaction for analytics"""
    ensure_directories()
    
    log_entry = {
        'timestamp': datetime.now().isoformat(),
        'emotion': emotion,
        'song_id': song_id,
        'reward': reward
    }
    
    log_file = 'logs/interactions.jsonl'
    
    try:
        with open(log_file, 'a') as f:
            f.write(json.dumps(log_entry) + '\n')
    except Exception as e:
        print(f"Error logging interaction: {e}")

def get_emotion_color(emotion: str) -> str:
    """Get color code for emotion"""
    from config.emotions import EMOTION_COLORS
    return EMOTION_COLORS.get(emotion, '#95A5A6')

def get_emotion_emoji(emotion: str) -> str:
    """Get emoji for emotion"""
    from config.emotions import EMOTION_EMOJIS
    return EMOTION_EMOJIS.get(emotion, '😐')

def format_song_data(songs: List[Dict]) -> List[Dict]:
    """Format song data for frontend"""
    formatted_songs = []
    
    for song in songs:
        formatted_song = {
            'id': song.get('id', ''),
            'title': song.get('title', 'Unknown'),
            'artist': song.get('artist', 'Unknown Artist'),
            'album': song.get('album', 'Unknown Album'),
            'preview_url': song.get('preview_url'),
            'image': song.get('image'),
            'spotify_url': song.get('spotify_url')
        }
        formatted_songs.append(formatted_song)
    
    return formatted_songs

def format_movie_data(movies: List[Dict]) -> List[Dict]:
    """Format movie data for frontend"""
    formatted_movies = []
    
    for movie in movies:
        formatted_movie = {
            'title': movie.get('title', 'Unknown'),
            'year': movie.get('year', 'N/A'),
            'poster_path': movie.get('poster_path'),
            'overview': movie.get('overview', ''),
            'rating': movie.get('rating', 0)
        }
        formatted_movies.append(formatted_movie)
    
    return formatted_movies

def calculate_session_stats(q_table: Dict) -> Dict:
    """Calculate statistics from Q-table"""
    total_emotions = len(q_table)
    total_songs = sum(len(songs) for songs in q_table.values())
    
    avg_q_values = {}
    for emotion, songs in q_table.items():
        if songs:
            avg_q_values[emotion] = sum(songs.values()) / len(songs)
    
    return {
        'total_emotions': total_emotions,
        'total_songs': total_songs,
        'avg_q_values': avg_q_values
    }

def validate_emotion(emotion: str) -> bool:
    """Validate if emotion is supported"""
    valid_emotions = ['happy', 'sad', 'angry', 'neutral', 'surprise', 'fear', 'disgust']
    return emotion.lower() in valid_emotions

def sanitize_filename(filename: str) -> str:
    """Sanitize filename for safe storage"""
    import re
    
    # Remove any non-alphanumeric characters except dots and underscores
    safe_name = re.sub(r'[^\w\s.-]', '', filename)
    safe_name = safe_name.replace(' ', '_')
    
    return safe_name

def get_timestamp() -> str:
    """Get current timestamp as string"""
    return datetime.now().strftime('%Y%m%d_%H%M%S')

def load_json_file(filepath: str) -> Dict:
    """Safely load JSON file"""
    try:
        if os.path.exists(filepath):
            with open(filepath, 'r') as f:
                return json.load(f)
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
    
    return {}

def save_json_file(filepath: str, data: Dict):
    """Safely save JSON file"""
    try:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
    except Exception as e:
        print(f"Error saving {filepath}: {e}")