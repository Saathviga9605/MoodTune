"""
Q-Learning Agent - FINAL VERSION
- No JSON tuple key errors
- No song/movie repeats
- Fresh start on new scan
- Like rate updates
"""

import json
import os
import random
from typing import List, Dict, Set

class QLearningAgent:
    def __init__(self, alpha=0.1, gamma=0.9, epsilon=0.2, q_table_path='data/q_table.json'):
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.q_table_path = q_table_path
        self.q_table = self._load_q_table()
        
        # Track shown items per (emotion, content_type)
        self.shown_history = {}  # {(emotion, content_type): set(indices)}
        
        self.stats = {
            'total_interactions': 0,
            'likes': 0,
            'dislikes': 0,
            'emotion_interactions': {},
            'content_stats': {'song': {}, 'movie': {}}
        }
        self._load_stats()
        self._load_shown_history()

    def _load_q_table(self) -> Dict:
        if os.path.exists(self.q_table_path):
            try:
                with open(self.q_table_path, 'r') as f:
                    loaded = json.load(f)
                self.q_table = {}
                for key, items in loaded.items():
                    emotion, content_type = key.rsplit('_', 1)
                    self.q_table[(emotion, content_type)] = {str(k): float(v) for k, v in items.items()}
                return self.q_table
            except Exception as e:
                print(f"Error loading Q-table: {e}")
        return {}

    def _save_q_table(self):
        try:
            os.makedirs(os.path.dirname(self.q_table_path), exist_ok=True)
            serializable = {}
            for (emotion, content_type), items in self.q_table.items():
                key = f"{emotion}_{content_type}"
                serializable[key] = {str(k): float(v) for k, v in items.items()}
            with open(self.q_table_path, 'w') as f:
                json.dump(serializable, f, indent=2)
        except Exception as e:
            print(f"Error saving Q-table: {e}")

    def _load_stats(self):
        stats_path = 'data/stats.json'
        if os.path.exists(stats_path):
            try:
                with open(stats_path, 'r') as f:
                    loaded = json.load(f)
                    self.stats.update(loaded)
            except Exception as e:
                print(f"Error loading stats: {e}")

    def _save_stats(self):
        try:
            os.makedirs('data', exist_ok=True)
            with open('data/stats.json', 'w') as f:
                json.dump(self.stats, f, indent=2)
        except Exception as e:
            print(f"Error saving stats: {e}")

    def _load_shown_history(self):
        path = 'data/shown_history.json'
        if os.path.exists(path):
            try:
                with open(path, 'r') as f:
                    loaded = json.load(f)
                self.shown_history = {}
                for key, indices in loaded.items():
                    emotion, content_type = key.rsplit('_', 1)
                    self.shown_history[(emotion, content_type)] = set(indices)
            except Exception as e:
                print(f"Error loading shown history: {e}")

    def _save_shown_history(self):
        try:
            os.makedirs('data', exist_ok=True)
            serializable = {}
            for (emotion, content_type), indices in self.shown_history.items():
                key = f"{emotion}_{content_type}"
                serializable[key] = list(indices)
            with open('data/shown_history.json', 'w') as f:
                json.dump(serializable, f, indent=2)
        except Exception as e:
            print(f"Error saving shown history: {e}")

    def mark_shown(self, emotion: str, content_type: str, index: int):
        key = (emotion.lower(), content_type.lower())
        if key not in self.shown_history:
            self.shown_history[key] = set()
        self.shown_history[key].add(index)
        self._save_shown_history()

    def reset_shown(self, emotion: str, content_type: str):
        key = (emotion.lower(), content_type.lower()) if emotion else None
        if key:
            self.shown_history.pop(key, None)
        else:
            # Reset all if no emotion
            self.shown_history = {}
        self._save_shown_history()

    def initialize_emotion(self, emotion: str, items: List[Dict], content_type: str = 'song'):
        emotion = emotion.lower()
        content_type = content_type.lower()
        key = (emotion, content_type)
        if key not in self.q_table:
            self.q_table[key] = {}
        id_field = 'id' if content_type == 'song' else 'title'
        for item in items:
            item_id = str(item.get(id_field) or item.get('name', hash(str(item))))
            if item_id not in self.q_table[key]:
                self.q_table[key][item_id] = 0.0
        if emotion not in self.stats['emotion_interactions']:
            self.stats['emotion_interactions'][emotion] = 0
        if emotion not in self.stats['content_stats'][content_type]:
            self.stats['content_stats'][content_type][emotion] = 0

    def select_action(self, emotion: str, items: List[Dict], content_type: str = 'song', exclude_indices: List[int] = None) -> int:
        emotion = emotion.lower()
        content_type = content_type.lower()
        key = (emotion, content_type)
        exclude_indices = exclude_indices or []

        # Initialize if needed
        if key not in self.q_table or not items:
            self.initialize_emotion(emotion, items, content_type)

        # Get shown history
        shown = self.shown_history.get(key, set())
        all_shown = len(shown) >= len(items)

        # Reset cycle if all shown
        if all_shown:
            self.reset_shown(emotion, content_type)
            shown = set()

        # Combine exclusions
        exclude = set(exclude_indices) | shown
        valid_items = [(i, item) for i, item in enumerate(items) if i not in exclude]

        if not valid_items:
            # Fallback: allow repeats only if no choice
            valid_items = [(i, item) for i, item in enumerate(items) if i not in exclude_indices]
            if not valid_items:
                return 0
            choice = random.choice(valid_items)
        else:
            id_field = 'id' if content_type == 'song' else 'title'
            if random.random() < self.epsilon:
                choice = random.choice(valid_items)
            else:
                choice = max(
                    valid_items,
                    key=lambda x: self.q_table[key].get(
                        str(x[1].get(id_field, hash(str(x[1])))), 0.0
                    )
                )

        selected_index = choice[0]
        self.mark_shown(emotion, content_type, selected_index)
        return selected_index

    def update_q_value(self, emotion: str, item_id: str, reward: float, content_type: str = 'song'):
        emotion = emotion.lower()
        content_type = content_type.lower()
        key = (emotion, content_type)
        if key not in self.q_table:
            self.q_table[key] = {}
        item_id = str(item_id)
        if item_id not in self.q_table[key]:
            self.q_table[key][item_id] = 0.0
        current_q = self.q_table[key][item_id]
        max_future_q = max(self.q_table[key].values()) if self.q_table[key] else 0.0
        new_q = current_q + self.alpha * (reward + self.gamma * max_future_q - current_q)
        self.q_table[key][item_id] = new_q

        self.stats['total_interactions'] += 1
        if reward > 0:
            self.stats['likes'] += 1
        else:
            self.stats['dislikes'] += 1
        self.stats['emotion_interactions'][emotion] = self.stats['emotion_interactions'].get(emotion, 0) + 1
        self.stats['content_stats'][content_type][emotion] = self.stats['content_stats'][content_type].get(emotion, 0) + 1
        
        self._save_q_table()
        self._save_stats()

    def get_stats(self) -> Dict:
        total = self.stats['total_interactions']
        like_rate = (self.stats['likes'] / total * 100) if total > 0 else 0
        return {
            'total_interactions': total,
            'like_rate': round(like_rate, 1),
            'likes': self.stats['likes'],
            'dislikes': self.stats['dislikes']
        }