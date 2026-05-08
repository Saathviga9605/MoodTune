const state = {
    currentView: 'home',
    emotion: null,
    confidence: 0,
    rawEmotion: null,
    rawConfidence: 0,
    emotionScores: {},
    songs: [],
    movies: [],
    mindfulness: [],
    currentSong: null,
    currentMovie: null,
    currentSongLoadedAt: 0,
    currentMovieLoadedAt: 0,
    stats: { total_interactions: 0, like_rate: 0 },
    analytics: {},
    detectionInterval: null
};

const emotionEmojis = {
    happy: '😊',
    sad: '😢',
    angry: '😠',
    neutral: '😐',
    surprise: '😲',
    fear: '😨',
    disgust: '🤢',
    calm: '🫧',
    relaxed: '😌',
    stressed: '😰'
};

const emotionOrder = ['happy', 'sad', 'angry', 'fear', 'surprise', 'neutral', 'disgust', 'calm', 'relaxed', 'stressed'];

const homeView = document.getElementById('home-view');
const scanningView = document.getElementById('scanning-view');
const resultsView = document.getElementById('results-view');
const startScanBtn = document.getElementById('start-scan-btn');
const resetBtn = document.getElementById('reset-btn');
const videoFeed = document.getElementById('video-feed');
const progressFrames = document.getElementById('progress-frames');
const progressPercent = document.getElementById('progress-percent');
const liveRawEmotion = document.getElementById('live-raw-emotion');
const liveRawConfidence = document.getElementById('live-raw-confidence');
const liveStableEmotion = document.getElementById('live-stable-emotion');
const liveStableConfidence = document.getElementById('live-stable-confidence');
const liveConfidenceBars = document.getElementById('live-confidence-bars');
const liveLatency = document.getElementById('live-latency');

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
    [homeView, scanningView, resultsView].forEach(view => view.classList.remove('active'));
    document.getElementById(`${viewName}-view`).classList.add('active');
    state.currentView = viewName;
}

function clamp01(value) {
    return Math.max(0, Math.min(1, value));
}

function renderConfidenceBars(scores = {}) {
    if (!liveConfidenceBars) return;
    liveConfidenceBars.innerHTML = '';
    emotionOrder.forEach(emotion => {
        const score = Number(scores[emotion] || 0);
        const row = document.createElement('div');
        row.className = 'confidence-row';
        row.innerHTML = `
            <div class="confidence-row-label">
                <span>${emotion}</span>
                <span>${Math.round(score * 100)}%</span>
            </div>
            <div class="confidence-track">
                <div class="confidence-fill ${emotion}" style="width:${clamp01(score) * 100}%"></div>
            </div>
        `;
        liveConfidenceBars.appendChild(row);
    });
}

function renderLiveAffect(data) {
    if (liveRawEmotion) liveRawEmotion.textContent = data.raw_emotion || 'Waiting...';
    if (liveRawConfidence) liveRawConfidence.textContent = `${Math.round((data.raw_confidence || 0) * 100)}%`;
    if (liveStableEmotion) liveStableEmotion.textContent = data.stabilized_emotion || data.emotion || 'Waiting...';
    if (liveStableConfidence) liveStableConfidence.textContent = `${Math.round((data.stabilized_confidence || data.confidence || 0) * 100)}%`;
    if (liveLatency) liveLatency.textContent = `${Math.round(data.latency_ms || 0)} ms`;
    renderConfidenceBars(data.emotion_scores || {});
}

async function handleStartScan() {
    showView('scanning');
    await fetch('/api/start-detection', { method: 'POST' });
    videoFeed.src = '/video_feed?' + Date.now();
    state.detectionInterval = setInterval(checkDetectionStatus, 350);
}

async function checkDetectionStatus() {
    try {
        const response = await fetch('/api/get-detection-status');
        const data = await response.json();
        progressFrames.textContent = data.frame_count || 0;
        progressPercent.textContent = Math.round(data.progress || 0);
        renderLiveAffect(data);

        if (data.complete) {
            clearInterval(state.detectionInterval);
            videoFeed.src = '';
            await fetch('/api/stop-detection', { method: 'POST' });
            state.emotion = data.stabilized_emotion || data.emotion;
            state.confidence = data.stabilized_confidence || data.confidence || 0;
            state.rawEmotion = data.raw_emotion || null;
            state.rawConfidence = data.raw_confidence || 0;
            await getRecommendations(state.emotion);
            showView('results');
        }
    } catch (err) {
        console.error('Detection error:', err);
    }
}

async function getRecommendations(emotion) {
    try {
        const response = await fetch('/api/get-recommendations', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ emotion })
        });
        const data = await response.json();

        if (data.success) {
            state.songs = data.songs || [];
            state.movies = data.movies || [];
            state.mindfulness = data.mindfulness || [];
            state.stats = data.rl_stats || state.stats;
            state.analytics = data.analytics || state.analytics;

            updateEmotionDisplay();
            updateStats();
            renderMindfulness();
            renderAnalytics(state.analytics);

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
    const emoji = emotionEmojis[state.emotion] || '😐';
    const emotionName = state.emotion ? state.emotion.charAt(0).toUpperCase() + state.emotion.slice(1) : 'Neutral';
    document.getElementById('emotion-emoji').textContent = emoji;
    document.getElementById('emotion-name').textContent = emotionName;
    document.getElementById('confidence-value').textContent = Math.round((state.confidence || 0) * 100);
}

function updateStats(stats = state.stats) {
    document.getElementById('total-interactions').textContent = stats.total_interactions || 0;
    document.getElementById('like-rate').textContent = `${stats.like_rate || 0}%`;
    document.getElementById('avg-reward').textContent = Number(stats.average_reward || 0).toFixed(3);
    document.getElementById('success-rate').textContent = `${Math.round(stats.success_rate || 0)}%`;
    document.getElementById('improvement-rate').textContent = `${Math.round(stats.emotional_improvement_percentage || 0)}%`;
    document.getElementById('convergence-score').textContent = Number(stats.convergence_score || 0).toFixed(3);
}

function renderAnalytics(analytics = {}) {
    const rewardTrend = document.getElementById('reward-trend-chart');
    const trajectory = document.getElementById('emotion-trajectory');
    const heatmap = document.getElementById('transition-heatmap');
    const distribution = document.getElementById('emotion-distribution');

    if (rewardTrend) {
        rewardTrend.innerHTML = '';
        (analytics.reward_trend || []).slice(-20).forEach(value => {
            const column = document.createElement('div');
            column.className = 'mini-bar';
            const normalized = clamp01((Number(value) + 1) / 2);
            column.style.height = `${Math.max(12, normalized * 100)}%`;
            column.title = Number(value).toFixed(3);
            rewardTrend.appendChild(column);
        });
    }

    if (trajectory) {
        const events = (analytics.transition_events || []).slice(-20);
        trajectory.innerHTML = events.length
            ? events.map(event => `<span class="trajectory-chip ${event.stabilized_emotion || event.next_emotion || 'neutral'}">${event.previous_emotion || event.raw_emotion || 'start'} → ${event.stabilized_emotion || event.next_emotion || 'neutral'}</span>`).join('')
            : '<span class="trajectory-empty">No trajectory data yet.</span>';
    }

    if (heatmap) {
        const labels = emotionOrder.slice(0, 7);
        const matrix = analytics.transition_summary || {};
        heatmap.innerHTML = '';
        const header = document.createElement('div');
        header.className = 'heatmap-row heatmap-header';
        header.innerHTML = '<span></span>' + labels.map(label => `<span>${label}</span>`).join('');
        heatmap.appendChild(header);
        labels.forEach(rowEmotion => {
            const row = document.createElement('div');
            row.className = 'heatmap-row';
            row.innerHTML = `<span class="heatmap-label">${rowEmotion}</span>` + labels.map(colEmotion => {
                const value = (((matrix[rowEmotion] || {})[colEmotion]) || 0);
                return `<div class="heatmap-cell" data-value="${value}">${value}</div>`;
            }).join('');
            heatmap.appendChild(row);
        });
    }

    if (distribution) {
        distribution.innerHTML = '';
        const dist = analytics.emotion_distribution || {};
        emotionOrder.slice(0, 7).forEach(emotion => {
            const value = Number(dist[emotion] || 0);
            const row = document.createElement('div');
            row.className = 'distribution-row';
            row.innerHTML = `
                <span>${emotion}</span>
                <div class="distribution-track">
                    <div class="distribution-fill ${emotion}" style="width:${Math.min(100, value * 12)}%"></div>
                </div>
                <strong>${value}</strong>
            `;
            distribution.appendChild(row);
        });
    }
}

function loadFirstSong() {
    if (state.songs.length === 0) return;
    state.currentSong = state.songs[0];
    state.currentSongLoadedAt = Date.now();
    updateSongCard();
}

function loadFirstMovie() {
    if (state.movies.length === 0) return;
    state.currentMovie = state.movies[0];
    state.currentMovieLoadedAt = Date.now();
    updateMovieCard();
}

function updateSongCard() {
    if (!state.currentSong) return;
    document.getElementById('song-title').textContent = state.currentSong.title;
    document.getElementById('song-artist').textContent = state.currentSong.artist;
    document.getElementById('song-album').textContent = state.currentSong.album;
    state.currentSongLoadedAt = Date.now();
}

function updateMovieCard() {
    if (!state.currentMovie) return;
    document.getElementById('movie-title').textContent = state.currentMovie.title;
    document.getElementById('movie-year').textContent = state.currentMovie.year;
    document.getElementById('movie-overview').textContent = state.currentMovie.overview || 'No overview available';
    state.currentMovieLoadedAt = Date.now();
}

function computeEngagementScore(loadedAt) {
    if (!loadedAt) return 0;
    const elapsedSeconds = (Date.now() - loadedAt) / 1000;
    return clamp01(elapsedSeconds / 20);
}

async function handleSongFeedback(liked) {
    if (!state.currentSong) return;

    songLikeBtn.disabled = true;
    songDislikeBtn.disabled = true;
    if (liked) {
        songLoveIndicator.classList.add('show');
        songCard.classList.add('swiping-right');
    } else {
        songPassIndicator.classList.add('show');
        songCard.classList.add('swiping-left');
    }

    const engagementScore = computeEngagementScore(state.currentSongLoadedAt);

    try {
        const response = await fetch('/api/feedback-song', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                emotion: state.emotion,
                song_id: state.currentSong.id,
                liked,
                engagement_score: engagementScore,
                completion_score: 1,
                mood_improvement_score: 0
            })
        });
        const data = await response.json();

        await new Promise(resolve => setTimeout(resolve, 400));
        songCard.classList.remove('swiping-left', 'swiping-right');
        songLoveIndicator.classList.remove('show');
        songPassIndicator.classList.remove('show');

        if (data.success && data.next_song) {
            state.currentSong = data.next_song;
            updateSongCard();
            updateStats(data.rl_stats);
            if (data.analytics) {
                state.analytics = data.analytics;
                renderAnalytics(state.analytics);
            }
        } else if (state.songs.length > 0) {
            const index = state.songs.findIndex(song => song.id === state.currentSong.id);
            const nextIndex = (index + 1) % state.songs.length;
            state.currentSong = state.songs[nextIndex];
            updateSongCard();
        }
    } catch (err) {
        console.error('Feedback failed:', err);
    } finally {
        songLikeBtn.disabled = false;
        songDislikeBtn.disabled = false;
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

    const engagementScore = computeEngagementScore(state.currentMovieLoadedAt);

    try {
        const response = await fetch('/api/feedback-movie', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                emotion: state.emotion,
                movie_title: state.currentMovie.title,
                liked,
                engagement_score: engagementScore,
                completion_score: 1,
                mood_improvement_score: 0
            })
        });
        const data = await response.json();

        await new Promise(resolve => setTimeout(resolve, 400));
        movieCard.classList.remove('swiping-left', 'swiping-right');
        movieLoveIndicator.classList.remove('show');
        moviePassIndicator.classList.remove('show');

        if (data.success && data.next_movie) {
            state.currentMovie = data.next_movie;
            updateMovieCard();
            updateStats(data.rl_stats);
            if (data.analytics) {
                state.analytics = data.analytics;
                renderAnalytics(state.analytics);
            }
        } else if (state.movies.length > 0) {
            const index = state.movies.findIndex(movie => movie.title === state.currentMovie.title);
            const nextIndex = (index + 1) % state.movies.length;
            state.currentMovie = state.movies[nextIndex];
            updateMovieCard();
        }
    } catch (err) {
        console.error('Movie feedback failed:', err);
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
    state.mindfulness.forEach((tip, index) => {
        const item = document.createElement('div');
        item.className = 'mindfulness-item';
        item.innerHTML = `
            <div class="mindfulness-number">${index + 1}</div>
            <p class="mindfulness-text">${tip}</p>
        `;
        list.appendChild(item);
    });
}

function setupTabs() {
    document.querySelectorAll('.tab').forEach(tab => {
        tab.addEventListener('click', () => {
            document.querySelectorAll('.tab').forEach(button => button.classList.remove('active'));
            document.querySelectorAll('.tab-content').forEach(panel => panel.classList.remove('active'));
            tab.classList.add('active');
            document.getElementById(`${tab.dataset.tab}-tab`).classList.add('active');
        });
    });
}

function handleReset() {
    if (state.detectionInterval) clearInterval(state.detectionInterval);
    fetch('/api/reset-session', { method: 'POST' });
    state.emotion = null;
    state.confidence = 0;
    state.rawEmotion = null;
    state.rawConfidence = 0;
    state.songs = [];
    state.movies = [];
    state.currentSong = null;
    state.currentMovie = null;
    state.analytics = {};
    videoFeed.src = '';
    showView('home');
}

startScanBtn.addEventListener('click', handleStartScan);
resetBtn.addEventListener('click', handleReset);
songLikeBtn.addEventListener('click', () => handleSongFeedback(true));
songDislikeBtn.addEventListener('click', () => handleSongFeedback(false));
movieLikeBtn.addEventListener('click', () => handleMovieFeedback(true));
movieDislikeBtn.addEventListener('click', () => handleMovieFeedback(false));

document.addEventListener('keydown', event => {
    if (state.currentView !== 'results') return;
    const activeTab = document.querySelector('.tab.active')?.dataset.tab;
    if (activeTab === 'songs') {
        if (event.key === 'ArrowLeft' || event.key === 'a') handleSongFeedback(false);
        if (event.key === 'ArrowRight' || event.key === 'd') handleSongFeedback(true);
    } else if (activeTab === 'movies') {
        if (event.key === 'ArrowLeft' || event.key === 'a') handleMovieFeedback(false);
        if (event.key === 'ArrowRight' || event.key === 'd') handleMovieFeedback(true);
    }
});

document.addEventListener('DOMContentLoaded', () => {
    setupTabs();
    renderConfidenceBars({});
    showView('home');
    console.log('MoodTune adaptive affective framework initialized');
});