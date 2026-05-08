"""
Emotion Configuration Module
Contains fallback data, color schemes, and emotion-specific content
"""

EMOTION_COLORS = {
    'happy': '#FFD700',
    'sad': '#4A90E2',
    'angry': '#E74C3C',
    'neutral': '#95A5A6',
    'surprise': '#F39C12',
    'fear': '#8E44AD',
    'disgust': '#16A085',
    'calm': '#2EC4B6',
    'relaxed': '#A8E6CF',
    'stressed': '#FF6F61'
}

EMOTION_EMOJIS = {
    'happy': '😊',
    'sad': '😢',
    'angry': '😠',
    'neutral': '😐',
    'surprise': '😲',
    'fear': '😨',
    'disgust': '🤢',
    'calm': '🫧',
    'relaxed': '😌',
    'stressed': '😰'
}

# Fallback songs when API fails
FALLBACK_SONGS = {
    'happy': [
        {'id': 'h1', 'title': 'Happy', 'artist': 'Pharrell Williams', 'album': 'Girl'},
        {'id': 'h2', 'title': 'Good Vibrations', 'artist': 'The Beach Boys', 'album': 'Smiley Smile'},
        {'id': 'h3', 'title': 'Walking on Sunshine', 'artist': 'Katrina and the Waves', 'album': 'Walking on Sunshine'},
        {'id': 'h4', 'title': "Don't Stop Me Now", 'artist': 'Queen', 'album': 'Jazz'},
        {'id': 'h5', 'title': 'I Got You (I Feel Good)', 'artist': 'James Brown', 'album': 'Out of Sight'},
        {'id': 'h6', 'title': 'September', 'artist': 'Earth, Wind & Fire', 'album': 'The Best of Earth, Wind & Fire'},
        {'id': 'h7', 'title': 'Best Day Of My Life', 'artist': 'American Authors', 'album': 'Oh, What a Life'},
        {'id': 'h8', 'title': 'Three Little Birds', 'artist': 'Bob Marley', 'album': 'Exodus'},
        {'id': 'h9', 'title': 'I Gotta Feeling', 'artist': 'The Black Eyed Peas', 'album': 'The E.N.D.'},
        {'id': 'h10', 'title': 'Here Comes The Sun', 'artist': 'The Beatles', 'album': 'Abbey Road'},
    ],
    'sad': [
        {'id': 's1', 'title': 'Someone Like You', 'artist': 'Adele', 'album': '21'},
        {'id': 's2', 'title': 'The Night We Met', 'artist': 'Lord Huron', 'album': 'Strange Trails'},
        {'id': 's3', 'title': 'Hurt', 'artist': 'Johnny Cash', 'album': 'American IV'},
        {'id': 's4', 'title': 'Mad World', 'artist': 'Gary Jules', 'album': 'Trading Snakeoil'},
        {'id': 's5', 'title': 'Tears in Heaven', 'artist': 'Eric Clapton', 'album': 'Unplugged'},
        {'id': 's6', 'title': 'Fix You', 'artist': 'Coldplay', 'album': 'X&Y'},
        {'id': 's7', 'title': 'The Scientist', 'artist': 'Coldplay', 'album': 'A Rush of Blood to the Head'},
        {'id': 's8', 'title': 'Skinny Love', 'artist': 'Bon Iver', 'album': 'For Emma, Forever Ago'},
        {'id': 's9', 'title': 'Hallelujah', 'artist': 'Jeff Buckley', 'album': 'Grace'},
        {'id': 's10', 'title': 'Nothing Compares 2 U', 'artist': 'Sinead O\'Connor', 'album': 'I Do Not Want What I Haven\'t Got'},
    ],
    'angry': [
        {'id': 'a1', 'title': 'Break Stuff', 'artist': 'Limp Bizkit', 'album': 'Significant Other'},
        {'id': 'a2', 'title': 'Killing in the Name', 'artist': 'Rage Against the Machine', 'album': 'Rage Against the Machine'},
        {'id': 'a3', 'title': 'Bodies', 'artist': 'Drowning Pool', 'album': 'Sinner'},
        {'id': 'a4', 'title': 'Last Resort', 'artist': 'Papa Roach', 'album': 'Infest'},
        {'id': 'a5', 'title': 'Down with the Sickness', 'artist': 'Disturbed', 'album': 'The Sickness'},
        {'id': 'a6', 'title': 'Chop Suey!', 'artist': 'System of a Down', 'album': 'Toxicity'},
        {'id': 'a7', 'title': 'Freak on a Leash', 'artist': 'Korn', 'album': 'Follow the Leader'},
        {'id': 'a8', 'title': 'Bulls on Parade', 'artist': 'Rage Against the Machine', 'album': 'Evil Empire'},
        {'id': 'a9', 'title': 'Enter Sandman', 'artist': 'Metallica', 'album': 'Metallica'},
        {'id': 'a10', 'title': 'Walk', 'artist': 'Pantera', 'album': 'Vulgar Display of Power'},
    ],
    'neutral': [
        {'id': 'n1', 'title': 'Weightless', 'artist': 'Marconi Union', 'album': 'Weightless'},
        {'id': 'n2', 'title': 'Clair de Lune', 'artist': 'Claude Debussy', 'album': 'Suite Bergamasque'},
        {'id': 'n3', 'title': 'Electric Feel', 'artist': 'MGMT', 'album': 'Oracular Spectacular'},
        {'id': 'n4', 'title': 'Let It Happen', 'artist': 'Tame Impala', 'album': 'Currents'},
        {'id': 'n5', 'title': 'Daydreaming', 'artist': 'Radiohead', 'album': 'A Moon Shaped Pool'},
        {'id': 'n6', 'title': 'Breathe', 'artist': 'Pink Floyd', 'album': 'The Dark Side of the Moon'},
        {'id': 'n7', 'title': 'Porcelain', 'artist': 'Moby', 'album': 'Play'},
        {'id': 'n8', 'title': 'Teardrop', 'artist': 'Massive Attack', 'album': 'Mezzanine'},
        {'id': 'n9', 'title': 'To Build a Home', 'artist': 'The Cinematic Orchestra', 'album': 'Ma Fleur'},
        {'id': 'n10', 'title': 'Holocene', 'artist': 'Bon Iver', 'album': 'Bon Iver'},
    ],
    'surprise': [
        {'id': 'sp1', 'title': 'Bohemian Rhapsody', 'artist': 'Queen', 'album': 'A Night at the Opera'},
        {'id': 'sp2', 'title': 'Mr. Blue Sky', 'artist': 'Electric Light Orchestra', 'album': 'Out of the Blue'},
        {'id': 'sp3', 'title': 'September', 'artist': 'Earth, Wind & Fire', 'album': 'The Best of Earth, Wind & Fire'},
        {'id': 'sp4', 'title': 'Superstition', 'artist': 'Stevie Wonder', 'album': 'Talking Book'},
        {'id': 'sp5', 'title': 'Uptown Funk', 'artist': 'Mark Ronson ft. Bruno Mars', 'album': 'Uptown Special'},
        {'id': 'sp6', 'title': 'Thriller', 'artist': 'Michael Jackson', 'album': 'Thriller'},
        {'id': 'sp7', 'title': 'Sweet Child O\' Mine', 'artist': 'Guns N\' Roses', 'album': 'Appetite for Destruction'},
        {'id': 'sp8', 'title': 'Livin\' on a Prayer', 'artist': 'Bon Jovi', 'album': 'Slippery When Wet'},
        {'id': 'sp9', 'title': 'Eye of the Tiger', 'artist': 'Survivor', 'album': 'Eye of the Tiger'},
        {'id': 'sp10', 'title': 'We Will Rock You', 'artist': 'Queen', 'album': 'News of the World'},
    ],
    'fear': [
        {'id': 'f1', 'title': 'Breathe Me', 'artist': 'Sia', 'album': 'Colour the Small One'},
        {'id': 'f2', 'title': 'Fix You', 'artist': 'Coldplay', 'album': 'X&Y'},
        {'id': 'f3', 'title': 'The Sound of Silence', 'artist': 'Simon & Garfunkel', 'album': 'Sounds of Silence'},
        {'id': 'f4', 'title': 'Brave', 'artist': 'Sara Bareilles', 'album': 'The Blessed Unrest'},
        {'id': 'f5', 'title': 'Unwritten', 'artist': 'Natasha Bedingfield', 'album': 'Unwritten'},
        {'id': 'f6', 'title': 'Fight Song', 'artist': 'Rachel Platten', 'album': 'Wildfire'},
        {'id': 'f7', 'title': 'Titanium', 'artist': 'David Guetta ft. Sia', 'album': 'Nothing but the Beat'},
        {'id': 'f8', 'title': 'Stronger', 'artist': 'Kelly Clarkson', 'album': 'Stronger'},
        {'id': 'f9', 'title': 'Rise Up', 'artist': 'Andra Day', 'album': 'Cheers to the Fall'},
        {'id': 'f10', 'title': 'Hall of Fame', 'artist': 'The Script ft. will.i.am', 'album': '#3'},
    ],
    'disgust': [
        {'id': 'd1', 'title': 'Shake It Off', 'artist': 'Taylor Swift', 'album': '1989'},
        {'id': 'd2', 'title': 'Confident', 'artist': 'Demi Lovato', 'album': 'Confident'},
        {'id': 'd3', 'title': 'Roar', 'artist': 'Katy Perry', 'album': 'Prism'},
        {'id': 'd4', 'title': 'Survivor', 'artist': 'Destiny\'s Child', 'album': 'Survivor'},
        {'id': 'd5', 'title': 'Stronger', 'artist': 'Kelly Clarkson', 'album': 'Stronger'},
        {'id': 'd6', 'title': 'Since U Been Gone', 'artist': 'Kelly Clarkson', 'album': 'Breakaway'},
        {'id': 'd7', 'title': 'Problem', 'artist': 'Ariana Grande', 'album': 'My Everything'},
        {'id': 'd8', 'title': 'Don\'t Stop Believin\'', 'artist': 'Journey', 'album': 'Escape'},
        {'id': 'd9', 'title': 'I Will Survive', 'artist': 'Gloria Gaynor', 'album': 'Love Tracks'},
        {'id': 'd10', 'title': 'Respect', 'artist': 'Aretha Franklin', 'album': 'I Never Loved a Man'},
    ]
}

FALLBACK_SONGS['calm'] = FALLBACK_SONGS['neutral']
FALLBACK_SONGS['relaxed'] = FALLBACK_SONGS['neutral']
FALLBACK_SONGS['stressed'] = FALLBACK_SONGS['neutral']

# Fallback movies
FALLBACK_MOVIES = {
    'happy': [
        {'title': 'The Grand Budapest Hotel', 'year': 2014, 'poster_path': '/cvhNhSP3vD9XZQaSYPdeme4SPbI.jpg'},
        {'title': 'Amélie', 'year': 2001, 'poster_path': '/nSxDa3M9aMvGVLoItzWTepQ5h5d.jpg'},
        {'title': 'Paddington 2', 'year': 2017, 'poster_path': '/gLGOXVRyb1MBmDh8tg2iXbJcfZz.jpg'},
        {'title': 'The Secret Life of Walter Mitty', 'year': 2013, 'poster_path': '/inC1KVw8BbPXnnht0d6YrqmI3Fx.jpg'},
    ],
    'sad': [
        {'title': 'Inside Out', 'year': 2015, 'poster_path': '/2H1TmgdfNtsKlU9jKdeNyYL5y8T.jpg'},
        {'title': 'The Pursuit of Happyness', 'year': 2006, 'poster_path': '/fPOJkeXhF4QRJyWk9rqzDwP5YTx.jpg'},
        {'title': 'Good Will Hunting', 'year': 1997, 'poster_path': '/bABCBKYBK7A5G1x0FzoeoNfuj2.jpg'},
        {'title': 'It\'s a Wonderful Life', 'year': 1946, 'poster_path': '/bSqt9rhDZx1Q7UZ86dBPKdNomp2.jpg'},
    ],
    'angry': [
        {'title': 'Fight Club', 'year': 1999, 'poster_path': '/pB8BM7pdSp6B6Ih7QZ4DrQ3PmJK.jpg'},
        {'title': 'Whiplash', 'year': 2014, 'poster_path': '/7fn624j5lj3xTme2SgiLCeuedmO.jpg'},
        {'title': 'Mad Max: Fury Road', 'year': 2015, 'poster_path': '/hA2ple9q4qnwxp3hKVNhroipsir.jpg'},
        {'title': 'John Wick', 'year': 2014, 'poster_path': '/fZPSd91yGE9fCcCe6OoQr6E3Bev.jpg'},
    ],
    'neutral': [
        {'title': 'Lost in Translation', 'year': 2003, 'poster_path': '/wV0SDQ8VBfPK6LfmEihXzuHapRm.jpg'},
        {'title': 'Her', 'year': 2013, 'poster_path': '/eCOtqtfvn7mxGl6nfmq4b1exJRc.jpg'},
        {'title': 'Paterson', 'year': 2016, 'poster_path': '/lXaKlpGPOuL17sjnNP2j5s0CZTQ.jpg'},
        {'title': 'Before Sunrise', 'year': 1995, 'poster_path': '/3V9yGOAADuNDKqVQx1pVVeVLJwE.jpg'},
    ],
    'surprise': [
        {'title': 'Inception', 'year': 2010, 'poster_path': '/9gk7adHYeDvHkCSEqAvQNLV5Uge.jpg'},
        {'title': 'The Prestige', 'year': 2006, 'poster_path': '/tRNlZbgNCNOpLpbPEz5L8G8A0JN.jpg'},
        {'title': 'Everything Everywhere All at Once', 'year': 2022, 'poster_path': '/w3LxiVYdWWRvEVdn5RYq6jIqkb1.jpg'},
        {'title': 'Arrival', 'year': 2016, 'poster_path': '/x2FJsf1ElAgr63Y3PNPtJrcmpoe.jpg'},
    ],
    'fear': [
        {'title': 'Finding Nemo', 'year': 2003, 'poster_path': '/eHuGQ10FUzK1mdOY69wF5pGgEf5.jpg'},
        {'title': 'The Pursuit of Happyness', 'year': 2006, 'poster_path': '/fPOJkeXhF4QRJyWk9rqzDwP5YTx.jpg'},
        {'title': 'Life of Pi', 'year': 2012, 'poster_path': '/8s8F0UUGjlLIyw0S8Nqtx2E1PzB.jpg'},
        {'title': 'Soul', 'year': 2020, 'poster_path': '/hm58Jw4Lw8OIeECIq5qyPYhAeRJ.jpg'},
    ],
    'disgust': [
        {'title': 'Inside Out', 'year': 2015, 'poster_path': '/2H1TmgdfNtsKlU9jKdeNyYL5y8T.jpg'},
        {'title': 'Ratatouille', 'year': 2007, 'poster_path': '/npHNjldbeTHdKKw28bJKs7lzqzj.jpg'},
        {'title': 'The Devil Wears Prada', 'year': 2006, 'poster_path': '/8912AsVuS7Sj915apArUFbv6F9L.jpg'},
        {'title': 'Mean Girls', 'year': 2004, 'poster_path': '/fXm3YKXAEjx7d2tIWDg9TfRZtsU.jpg'},
    ]
}

FALLBACK_MOVIES['calm'] = FALLBACK_MOVIES['neutral']
FALLBACK_MOVIES['relaxed'] = FALLBACK_MOVIES['neutral']
FALLBACK_MOVIES['stressed'] = FALLBACK_MOVIES['neutral']

# Mindfulness tips for each emotion
MINDFULNESS_TIPS = {
    'happy': [
        'Share your joy with someone today',
        'Practice gratitude by listing 3 things you\'re thankful for',
        'Engage in a creative activity you love',
        'Spend time in nature and appreciate its beauty',
        'Celebrate small wins and acknowledge your achievements'
    ],
    'sad': [
        'Allow yourself to feel your emotions without judgment',
        'Take 5 deep breaths: inhale for 4, hold for 4, exhale for 6',
        'Reach out to a trusted friend or family member',
        'Write down your feelings in a journal',
        'Remember: This too shall pass. Emotions are temporary'
    ],
    'angry': [
        'Practice progressive muscle relaxation',
        'Go for a vigorous walk or run',
        'Channel energy into physical activity',
        'Count backwards from 100 by 7s to reset your mind',
        'Take a timeout before responding to triggering situations'
    ],
    'neutral': [
        'Try a 10-minute guided meditation',
        'Explore a new hobby or interest',
        'Practice mindful observation of your surroundings',
        'Set an intention for the rest of your day',
        'Engage in light stretching or yoga'
    ],
    'surprise': [
        'Embrace the unexpected with curiosity',
        'Journal about what surprised you today',
        'Try something completely new',
        'Share your surprise with someone close to you',
        'Reflect on how surprises lead to growth'
    ],
    'fear': [
        'Practice the 5-4-3-2-1 grounding technique',
        'Remind yourself: "This too shall pass"',
        'Connect with your support system',
        'Take slow, deep belly breaths for 2 minutes',
        'Name your fear and acknowledge it without judgment'
    ],
    'disgust': [
        'Acknowledge and validate your feelings',
        'Cleanse your space - physical tidying can help mental clarity',
        'Take a refreshing shower or wash your face',
        'Practice self-compassion and kindness',
        'Distance yourself from the source if possible'
    ],
    'calm': [
        'Let this steadiness support the rest of your day',
        'Notice the calmness in your breathing',
        'Use this moment to plan one small meaningful task',
        'Stay with the sensation of ease for a few breaths',
        'Protect the calm by limiting unnecessary stimulation'
    ],
    'relaxed': [
        'Let the relaxation settle fully into your body',
        'Enjoy the slower pace without forcing the next task',
        'Extend this feeling with soft music or a quiet walk',
        'Notice where your muscles have softened',
        'Use this state to recharge before the next challenge'
    ],
    'stressed': [
        'Reduce input and simplify your next action',
        'Take 10 slow breaths and lengthen your exhale',
        'Name the most urgent task and ignore the rest for now',
        'Relax your shoulders and unclench your jaw',
        'Swap urgency for a single manageable step'
    ]
}

# Emotion query keywords for Spotify
EMOTION_MUSIC_QUERIES = {
    'happy': ['happy', 'upbeat', 'cheerful', 'joyful', 'energetic'],
    'sad': ['sad', 'melancholy', 'emotional', 'heartbreak', 'blues'],
    'angry': ['angry', 'aggressive', 'metal', 'rock', 'intense'],
    'neutral': ['ambient', 'chill', 'calm', 'peaceful', 'relaxing'],
    'surprise': ['upbeat', 'energetic', 'exciting', 'dynamic', 'fun'],
    'fear': ['calming', 'soothing', 'peaceful', 'comforting', 'hopeful'],
    'disgust': ['empowering', 'strong', 'confident', 'motivational', 'uplifting']
}

EMOTION_MUSIC_QUERIES['calm'] = ['calm', 'soothing', 'ambient', 'relaxing', 'peaceful']
EMOTION_MUSIC_QUERIES['relaxed'] = ['relaxing', 'acoustic', 'ambient', 'soft', 'peaceful']
EMOTION_MUSIC_QUERIES['stressed'] = ['calming', 'stress relief', 'lofi', 'peaceful', 'breathing']

# Movie genres by emotion
EMOTION_MOVIE_GENRES = {
    'happy': ['comedy', 'family', 'romance', 'animation'],
    'sad': ['drama', 'romance'],
    'angry': ['action', 'thriller'],
    'neutral': ['documentary', 'drama'],
    'surprise': ['mystery', 'thriller', 'science fiction'],
    'fear': ['animation', 'family', 'adventure'],
    'disgust': ['comedy', 'drama']
}

EMOTION_MOVIE_GENRES['calm'] = ['documentary', 'drama']
EMOTION_MOVIE_GENRES['relaxed'] = ['drama', 'family']
EMOTION_MOVIE_GENRES['stressed'] = ['documentary', 'family']