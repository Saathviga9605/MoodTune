// MoodTune Frontend - FINAL, FLAWLESS, ZADE-APPROVED

const state = {
    currentView: 'home',
    emotion: null,
    confidence: 0,
    songs: [],
    movies: [],
    mindfulness: [],
    currentSong: null,
    currentMovie: null,
    stats: { total_interactions: 0, like_rate: 0 },
    detectionInterval: null
};

const emotionEmojis = {
    'happy': '😊', 'sad': '😢', 'angry': '😠', 'neutral': '😐',
    'surprise': '😲', 'fear': '😨', 'disgust': '🤢'
};

// DOM Elements
const homeView = document.getElementById('home-view');
const scanningView = document.getElementById('scanning-view');
const resultsView = document.getElementById('results-view');
const startScanBtn = document.getElementById('start-scan-btn');
const resetBtn = document.getElementById('reset-btn');
const videoFeed = document.getElementById('video-feed');
const progressFrames = document.getElementById('progress-frames');
const progressPercent = document.getElementById('progress-percent');

const songCard = document.getElementById('song-card');
const songLikeBtn = document.getElementById('song-like-btn');
const songDislikeBtn = document.getElementById('song-dislike-btn');
const songLoveIndicator = document.getElementById('song-love-indicator');
const songPassIndicator = document.getElementById('song-pass-indicator');

const movieCard = document.getElementById('movie-card');
const movieLikeBtn = document.getElementById('movie-like-btn');
const movieDislikeBtn = document.getElementById('movie-dislike-btn');
const movieLoveIndicator = document.getElementById('movie-love-indicator');
const moviePassIndicator = document.getElementById('movie-pass-indicator');

function showView(viewName) {
    [homeView, scanningView, resultsView].forEach(v => v.classList.remove('active'));
    document.getElementById(`${viewName}-view`).classList.add('active');
    state.currentView = viewName;
}

async function handleStartScan() {
    showView('scanning');
    await fetch('/api/start-detection', { method: 'POST' });
    videoFeed.src = '/video_feed?' + new Date().getTime();
    state.detectionInterval = setInterval(checkDetectionStatus, 500);
}

async function checkDetectionStatus() {
    try {
        const res = await fetch('/api/get-detection-status');
        const data = await res.json();
        progressFrames.textContent = data.frame_count || 0;
        progressPercent.textContent = Math.round(data.progress || 0);
        if (data.complete) {
            clearInterval(state.detectionInterval);
            videoFeed.src = '';
            await fetch('/api/stop-detection', { method: 'POST' });
            state.emotion = data.emotion;
            state.confidence = data.confidence;
            await getRecommendations(state.emotion);
            showView('results');
        }
    } catch (err) {
        console.error('Detection error:', err);
    }
}

async function getRecommendations(emotion) {
    try {
        const res = await fetch('/api/get-recommendations', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ emotion })
        });
        const data = await res.json();

        if (data.success) {
            state.songs = data.songs;
            state.movies = data.movies;
            state.mindfulness = data.mindfulness;
            state.stats = data.rl_stats;

            updateEmotionDisplay();
            updateStats();
            renderMindfulness();

            // NOW SAFE: Load first items directly
            loadFirstSong();
            loadFirstMovie();
        } else {
            alert('Failed to load recommendations. Try again.');
            showView('home');
        }
    } catch (err) {
        console.error('Failed to get recommendations:', err);
        alert('Connection error. Please try again.');
        showView('home');
    }
}

function updateEmotionDisplay() {
    document.getElementById('emotion-emoji').textContent = emotionEmojis[state.emotion] || '😐';
    document.getElementById('emotion-name').textContent = 
        state.emotion.charAt(0).toUpperCase() + state.emotion.slice(1);
    document.getElementById('confidence-value').textContent = Math.round(state.confidence * 100);
}

function updateStats(stats = state.stats) {
    document.getElementById('total-interactions').textContent = stats.total_interactions || 0;
    document.getElementById('like-rate').textContent = (stats.like_rate || 0) + '%';
}

// LOAD FIRST SONG — NO API CALL
function loadFirstSong() {
    if (state.songs.length === 0) return;
    state.currentSong = state.songs[0];
    updateSongCard();
}

// LOAD FIRST MOVIE — NO API CALL
function loadFirstMovie() {
    if (state.movies.length === 0) return;
    state.currentMovie = state.movies[0];
    updateMovieCard();
}

function updateSongCard() {
    if (!state.currentSong) return;
    document.getElementById('song-title').textContent = state.currentSong.title;
    document.getElementById('song-artist').textContent = state.currentSong.artist;
    document.getElementById('song-album').textContent = state.currentSong.album;
}

function updateMovieCard() {
    if (!state.currentMovie) return;
    document.getElementById('movie-title').textContent = state.currentMovie.title;
    document.getElementById('movie-year').textContent = state.currentMovie.year;
    document.getElementById('movie-overview').textContent = 
        state.currentMovie.overview || 'No overview available';
}

// SONG FEEDBACK — ONLY HERE DO WE CALL /api/next-song
async function handleSongFeedback(liked) {
    if (!state.currentSong) return;

    songLikeBtn.disabled = true; songDislikeBtn.disabled = true;
    if (liked) {
        songLoveIndicator.classList.add('show');
        songCard.classList.add('swiping-right');
    } else {
        songPassIndicator.classList.add('show');
        songCard.classList.add('swiping-left');
    }

    try {
        const res = await fetch('/api/feedback-song', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                emotion: state.emotion,
                song_id: state.currentSong.id,
                liked: liked
            })
        });
        const data = await res.json();

        await new Promise(r => setTimeout(r, 400));
        songCard.classList.remove('swiping-left', 'swiping-right');
        songLoveIndicator.classList.remove('show');
        songPassIndicator.classList.remove('show');

        if (data.success && data.next_song) {
            state.currentSong = data.next_song;
            updateSongCard();
            updateStats(data.rl_stats);
        } else {
            // Fallback
            const idx = state.songs.findIndex(s => s.id === state.currentSong.id);
            const nextIdx = (idx + 1) % state.songs.length;
            state.currentSong = state.songs[nextIdx];
            updateSongCard();
        }
    } catch (err) {
        console.error('Feedback failed:', err);
    } finally {
        songLikeBtn.disabled = false; songDislikeBtn.disabled = false;
    }
}

async function handleMovieFeedback(liked) {
    if (!state.currentMovie) return;

    movieLikeBtn.disabled = true;
    movieDislikeBtn.disabled = true;

    if (liked) {
        movieLoveIndicator.classList.add('show');
        movieCard.classList.add('swiping-right');
    } else {
        moviePassIndicator.classList.add('show');
        movieCard.classList.add('swiping-left');
    }

    try {
        const res = await fetch('/api/feedback-movie', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                emotion: state.emotion,
                movie_title: state.currentMovie.title,
                liked: liked
            })
        });
        const data = await res.json();

        await new Promise(r => setTimeout(r, 400));
        movieCard.classList.remove('swiping-left', 'swiping-right');
        movieLoveIndicator.classList.remove('show');
        moviePassIndicator.classList.remove('show');

        if (data.success && data.next_movie) {
            state.currentMovie = data.next_movie;
            updateMovieCard();
            updateStats(data.rl_stats);
        } else {
            // Fallback: pick next movie from list
            const idx = state.movies.findIndex(m => m.title === state.currentMovie.title);
            const nextIdx = (idx + 1) % state.movies.length;
            state.currentMovie = state.movies[nextIdx];
            updateMovieCard();
        }
    } catch (err) {
        console.error('Movie feedback failed:', err);
        // Fallback even if API fails
        const idx = state.movies.findIndex(m => m.title === state.currentMovie.title);
        const nextIdx = (idx + 1) % state.movies.length;
        state.currentMovie = state.movies[nextIdx];
        updateMovieCard();
    } finally {
        movieLikeBtn.disabled = false;
        movieDislikeBtn.disabled = false;
    }
}

function renderMindfulness() {
    const list = document.getElementById('mindfulness-list');
    list.innerHTML = '';
    if (state.mindfulness.length === 0) {
        list.innerHTML = '<p class="mindfulness-text">No tips available.</p>';
        return;
    }
    state.mindfulness.forEach((tip, i) => {
        const item = document.createElement('div');
        item.className = 'mindfulness-item';
        item.innerHTML = `
            <div class="mindfulness-number">${i + 1}</div>
            <p class="mindfulness-text">${tip}</p>
        `;
        list.appendChild(item);
    });
}

function setupTabs() {
    document.querySelectorAll('.tab').forEach(tab => {
        tab.addEventListener('click', () => {
            document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
            document.querySelectorAll('.tab-content').forEach(tc => tc.classList.remove('active'));
            tab.classList.add('active');
            document.getElementById(`${tab.dataset.tab}-tab`).classList.add('active');
        });
    });
}

function handleReset() {
    if (state.detectionInterval) clearInterval(state.detectionInterval);
    fetch('/api/reset-session', { method: 'POST' });
    state.emotion = null;
    state.songs = [];
    state.movies = [];
    state.currentSong = null;
    state.currentMovie = null;
    videoFeed.src = '';
    showView('home');
}

// EVENT LISTENERS
startScanBtn.addEventListener('click', handleStartScan);
resetBtn.addEventListener('click', handleReset);

songLikeBtn.addEventListener('click', () => handleSongFeedback(true));
songDislikeBtn.addEventListener('click', () => handleSongFeedback(false));

movieLikeBtn.addEventListener('click', () => handleMovieFeedback(true));
movieDislikeBtn.addEventListener('click', () => handleMovieFeedback(false));

// KEYBOARD SHORTCUTS
document.addEventListener('keydown', e => {
    if (state.currentView !== 'results') return;
    const activeTab = document.querySelector('.tab.active')?.dataset.tab;
    if (activeTab === 'songs') {
        if (e.key === 'ArrowLeft' || e.key === 'a') handleSongFeedback(false);
        if (e.key === 'ArrowRight' || e.key === 'd') handleSongFeedback(true);
    } else if (activeTab === 'movies') {
        if (e.key === 'ArrowLeft' || e.key === 'a') handleMovieFeedback(false);
        if (e.key === 'ArrowRight' || e.key === 'd') handleMovieFeedback(true);
    }
});

// INIT
document.addEventListener('DOMContentLoaded', () => {
    setupTabs();
    showView('home');
    console.log('MoodTune v2.0 - Zade-Approved, 400-Proof');
});