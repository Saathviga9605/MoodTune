"""
API Configuration Module
Handles Spotify and TMDB API integrations
"""

import os
import time
import requests
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
from dotenv import load_dotenv
from config.emotions import EMOTION_MUSIC_QUERIES, EMOTION_MOVIE_GENRES, FALLBACK_SONGS, FALLBACK_MOVIES

load_dotenv()

class SpotifyAPI:
    def __init__(self):
        self.client_id = os.getenv('SPOTIPY_CLIENT_ID')
        self.client_secret = os.getenv('SPOTIPY_CLIENT_SECRET')
        self.sp = None
        self.cache = {}
        self.cache_ttl = 300
        
        if self.client_id and self.client_secret:
            try:
                auth_manager = SpotifyClientCredentials(
                    client_id=self.client_id,
                    client_secret=self.client_secret
                )
                self.sp = spotipy.Spotify(auth_manager=auth_manager)
            except Exception as e:
                print(f"Spotify API initialization failed: {e}")
    
    def search_songs_by_emotion(self, emotion, limit=15):
        """Search for songs based on emotion"""
        cache_key = (emotion.lower(), int(limit))
        cached = self.cache.get(cache_key)
        if cached and (time.time() - cached['timestamp'] < self.cache_ttl):
            return cached['data']

        if not self.sp:
            print("Spotify API not initialized, using fallback")
            songs = FALLBACK_SONGS.get(emotion, [])
            self.cache[cache_key] = {'timestamp': time.time(), 'data': songs[:limit]}
            return songs[:limit]
        
        try:
            queries = EMOTION_MUSIC_QUERIES.get(emotion, ['music'])
            all_songs = []
            
            for query in queries[:3]:  # Use first 3 queries
                results = self.sp.search(q=query, type='track', limit=5)
                
                for track in results['tracks']['items']:
                    song_data = {
                        'id': track['id'],
                        'title': track['name'],
                        'artist': track['artists'][0]['name'],
                        'album': track['album']['name'],
                        'preview_url': track.get('preview_url'),
                        'image': track['album']['images'][0]['url'] if track['album']['images'] else None,
                        'spotify_url': track['external_urls']['spotify']
                    }
                    all_songs.append(song_data)
            
            songs = all_songs[:limit]
            self.cache[cache_key] = {'timestamp': time.time(), 'data': songs}
            return songs
        
        except Exception as e:
            print(f"Spotify search failed: {e}")
            songs = FALLBACK_SONGS.get(emotion, [])
            self.cache[cache_key] = {'timestamp': time.time(), 'data': songs[:limit]}
            return songs[:limit]


class TMDBAPI:
    def __init__(self):
        self.api_key = os.getenv('TMDB_API_KEY')
        self.base_url = 'https://api.themoviedb.org/3'
        self.image_base_url = 'https://image.tmdb.org/t/p/w500'
        self.cache = {}
        self.cache_ttl = 300
    
    def search_movies_by_emotion(self, emotion, limit=4):
        """Search for movies based on emotion"""
        cache_key = (emotion.lower(), int(limit))
        cached = self.cache.get(cache_key)
        if cached and (time.time() - cached['timestamp'] < self.cache_ttl):
            return cached['data']

        if not self.api_key:
            print("TMDB API key not found, using fallback")
            movies = FALLBACK_MOVIES.get(emotion, [])
            self.cache[cache_key] = {'timestamp': time.time(), 'data': movies[:limit]}
            return movies[:limit]
        
        try:
            genres = EMOTION_MOVIE_GENRES.get(emotion, ['drama'])
            genre_ids = self._get_genre_ids(genres)
            
            # Discover movies with genre
            endpoint = f"{self.base_url}/discover/movie"
            params = {
                'api_key': self.api_key,
                'with_genres': '|'.join(map(str, genre_ids)),
                'sort_by': 'popularity.desc',
                'vote_average.gte': 7,
                'page': 1
            }
            
            response = requests.get(endpoint, params=params, timeout=5)
            response.raise_for_status()
            data = response.json()
            
            movies = []
            for movie in data.get('results', [])[:limit]:
                movie_data = {
                    'title': movie['title'],
                    'year': movie.get('release_date', '')[:4],
                    'poster_path': f"{self.image_base_url}{movie['poster_path']}" if movie.get('poster_path') else None,
                    'overview': movie.get('overview', ''),
                    'rating': movie.get('vote_average', 0)
                }
                movies.append(movie_data)
            
            resolved = movies if movies else FALLBACK_MOVIES.get(emotion, [])
            self.cache[cache_key] = {'timestamp': time.time(), 'data': resolved[:limit]}
            return resolved[:limit]
        
        except Exception as e:
            print(f"TMDB search failed: {e}")
            movies = FALLBACK_MOVIES.get(emotion, [])
            self.cache[cache_key] = {'timestamp': time.time(), 'data': movies[:limit]}
            return movies[:limit]
    
    def _get_genre_ids(self, genre_names):
        """Get genre IDs from genre names"""
        # Common genre mappings
        genre_map = {
            'action': 28,
            'adventure': 12,
            'animation': 16,
            'comedy': 35,
            'crime': 80,
            'documentary': 99,
            'drama': 18,
            'family': 10751,
            'fantasy': 14,
            'history': 36,
            'horror': 27,
            'music': 10402,
            'mystery': 9648,
            'romance': 10749,
            'science fiction': 878,
            'thriller': 53,
            'war': 10752,
            'western': 37
        }
        
        return [genre_map.get(genre, 18) for genre in genre_names]