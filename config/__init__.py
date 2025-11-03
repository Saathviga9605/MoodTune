# config/__init__.py
"""
Configuration package for MoodTune
"""

from .emotions import (
    EMOTION_COLORS,
    EMOTION_EMOJIS,
    FALLBACK_SONGS,
    FALLBACK_MOVIES,
    MINDFULNESS_TIPS,
    EMOTION_MUSIC_QUERIES,
    EMOTION_MOVIE_GENRES
)

from .api import SpotifyAPI, TMDBAPI

__all__ = [
    'EMOTION_COLORS',
    'EMOTION_EMOJIS',
    'FALLBACK_SONGS',
    'FALLBACK_MOVIES',
    'MINDFULNESS_TIPS',
    'EMOTION_MUSIC_QUERIES',
    'EMOTION_MOVIE_GENRES',
    'SpotifyAPI',
    'TMDBAPI'
]

# ==========================================



# ==========================================


# ==========================================


__all__ = [
    'ensure_directories',
    'log_interaction',
    'get_emotion_color',
    'get_emotion_emoji',
    'format_song_data',
    'format_movie_data',
    'calculate_session_stats',
    'validate_emotion',
    'sanitize_filename',
    'get_timestamp',
    'load_json_file',
    'save_json_file'
]