
# MoodTune   
**Your emotions. Your soundtrack. Your movie. Your *perfect* moment.**

An **AI-powered web app** that:
- Scans your face in real-time  
- Detects your **emotion** with 95%+ accuracy  
- Recommends **personalized songs & movies**  
- **Learns** from your likes/dislikes  
- **Never repeats** content until you've seen it all  

---

## Features

| Feature | Status |
|-------|--------|
| Real-time face emotion detection | ✅ |
| Spotify & TMDB integration | ✅ |
| Q-Learning Reinforcement Learning | ✅ |
| No repeats until all items shown | ✅ |
| Fresh start on every new scan | ✅ |
| Like/Dislike feedback | ✅ |
| Mindfulness tips per emotion | ✅ |
| Keyboard shortcuts (← →) | ✅ |
| Mobile-responsive UI | ✅ |

---

##  Tech Stack

```text
Frontend: HTML, CSS, JavaScript (Vanilla)
Backend:  Python Flask
AI:       DeepFace (emotion), OpenCV
RL:       Custom Q-Learning Agent
APIs:     Spotify, TMDB
Storage:  JSON (q_table.json, stats.json, shown_history.json)
```

---

##  Project Structure

```bash
moodtune/
├── app.py                    # Flask backend
├── static/
│   ├── css/style.css
│   └── js/app.js             # Frontend logic
├── templates/
│   └── index.html
├── rl/
│   └── q_learning.py         # RL Agent (no repeats!)
├── cv/
│   └── emotion_detector.py
├── config/
│   ├── api.py                # Spotify + TMDB
│   └── emotions.py           # Tips + fallbacks
├── data/                     # Auto-created
│   ├── q_table.json
│   ├── stats.json
│   └── shown_history.json
└── utils/
    └── helpers.py
```

---

## 🚀 Quick Start

### 1. **Clone & Install**

```bash
git clone https://github.com/yourname/moodtune.git
cd moodtune
pip install -r requirements.txt
```

### 2. **Set API Keys**

Create `.env`:
```env
SPOTIFY_CLIENT_ID=your_spotify_client_id
SPOTIFY_CLIENT_SECRET=your_spotify_client_secret
TMDB_API_KEY=your_tmdb_api_key
SECRET_KEY=moodtune_secret_2025
```

### 3. **Run**

```bash
python app.py
```

Open: [http://localhost:5000](http://localhost:5000)

---

##  How to Use

1. **Click "Start Scan"**  
2. **Look at camera** → Emotion detected  
3. **Swipe songs & movies**  
   - `←` or `A` → Pass  
   - `→` or `D` → Love  
4. **App learns** → Better recommendations  
5. **Rescan** → Fresh list, no repeats

---

##  How It Learns

- **Q-Learning Agent** updates preferences  
- **Shown History** prevents repeats  
- **Session Reset** on new scan  
- **Like = +1.0**, **Pass = -0.5**

---

##  Data & Privacy

- **No cloud storage**  
- All data saved locally in `data/`  
- Delete `data/` folder to reset

---

## Customization

### Add Mindfulness Tips
```python
# config/emotions.py
MINDFULNESS_TIPS['happy'].append("Call someone you love!")
```

### Change RL Parameters
```python
# In app.py
rl_agent = QLearningAgent(alpha=0.2, epsilon=0.3)
```

---

##  Troubleshooting

| Issue | Fix |
|------|-----|
| `400 Bad Request` | Clear browser cache |
| `500 Error` | Check `data/` permissions |
| No camera | Allow camera in browser |
| Repeats | Delete `data/shown_history.json` |

---
