"""
Adaptive Q-Learning Agent with multi-factor reward modeling and emotion transition analytics.
"""

import json
import os
import random
import time
from collections import defaultdict, deque
from datetime import datetime
from typing import List, Dict, Optional

from utils.helpers import log_event


class QLearningAgent:
    def __init__(self, alpha=0.1, gamma=0.9, epsilon=0.2, q_table_path='data/q_table.json'):
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.q_table_path = q_table_path
        self.reward_weights = {
            'feedback_score': 0.40,
            'engagement_score': 0.20,
            'mood_improvement_score': 0.30,
            'completion_score': 0.10,
        }
        self.q_table = self._load_q_table()

        self.shown_history = {}
        self.stats = {
            'total_interactions': 0,
            'likes': 0,
            'dislikes': 0,
            'emotion_interactions': {},
            'content_stats': {'song': {}, 'movie': {}},
        }

        self.reward_history_path = 'data/reward_history.json'
        self.transition_path = 'data/emotion_transitions.json'
        self.outcome_path = 'data/recommendation_outcomes.json'

        self.reward_history = self._load_json_list(self.reward_history_path)
        self.recommendation_outcomes = self._load_json_list(self.outcome_path)
        self.transition_matrix = self._load_transition_matrix()
        self.transition_history = deque(maxlen=20)
        self.convergence_history = deque(maxlen=100)

        self._load_stats()
        self._load_shown_history()

        self.rl_training_time = 0.0

        self.transition_scores = {
            'sad': {'neutral': 0.55, 'calm': 0.65, 'relaxed': 0.75, 'happy': 0.9, 'angry': -0.7, 'stressed': -0.5},
            'angry': {'calm': 0.9, 'relaxed': 0.8, 'neutral': 0.45, 'happy': 0.75, 'sad': -0.45, 'stressed': -0.35},
            'stressed': {'relaxed': 0.9, 'calm': 0.8, 'neutral': 0.5, 'happy': 0.7, 'angry': -0.6, 'sad': -0.4},
            'fear': {'calm': 0.8, 'relaxed': 0.75, 'neutral': 0.45, 'happy': 0.7, 'angry': -0.5},
            'disgust': {'neutral': 0.35, 'calm': 0.45, 'relaxed': 0.55, 'happy': 0.6, 'angry': -0.4},
            'neutral': {'happy': 0.45, 'calm': 0.35, 'relaxed': 0.4, 'stressed': -0.2},
            'happy': {'happy': 0.15, 'neutral': 0.1, 'calm': 0.2, 'relaxed': 0.2, 'angry': -0.85, 'sad': -0.35},
            'calm': {'relaxed': 0.55, 'happy': 0.35, 'neutral': 0.2, 'stressed': -0.3},
            'relaxed': {'happy': 0.3, 'neutral': 0.15, 'calm': 0.25, 'stressed': -0.25},
        }

    # ---------------------------
    # Data Loading and Saving
    # ---------------------------

    def _load_json_list(self, path: str) -> List[Dict]:
        if os.path.exists(path):
            try:
                with open(path, 'r') as file_handle:
                    loaded = json.load(file_handle)
                    return loaded if isinstance(loaded, list) else []
            except Exception as e:
                print(f"Error loading {path}: {e}")
        return []

    def _save_json_list(self, path: str, data: List[Dict]):
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, 'w') as file_handle:
                json.dump(data[-1000:], file_handle, indent=2)
        except Exception as e:
            print(f"Error saving {path}: {e}")

    def _load_q_table(self) -> Dict:
        if os.path.exists(self.q_table_path):
            try:
                with open(self.q_table_path, 'r') as file_handle:
                    loaded = json.load(file_handle)
                q_table = {}
                for key, items in loaded.items():
                    emotion, content_type = key.rsplit('_', 1)
                    q_table[(emotion, content_type)] = {str(k): float(v) for k, v in items.items()}
                return q_table
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
            with open(self.q_table_path, 'w') as file_handle:
                json.dump(serializable, file_handle, indent=2)
        except Exception as e:
            print(f"Error saving Q-table: {e}")

    def _load_stats(self):
        stats_path = 'data/stats.json'
        if os.path.exists(stats_path):
            try:
                with open(stats_path, 'r') as file_handle:
                    loaded = json.load(file_handle)
                    self.stats.update(loaded)
            except Exception as e:
                print(f"Error loading stats: {e}")

    def _save_stats(self):
        try:
            os.makedirs('data', exist_ok=True)
            with open('data/stats.json', 'w') as file_handle:
                json.dump(self.stats, file_handle, indent=2)
        except Exception as e:
            print(f"Error saving stats: {e}")

    def _load_shown_history(self):
        path = 'data/shown_history.json'
        if os.path.exists(path):
            try:
                with open(path, 'r') as file_handle:
                    loaded = json.load(file_handle)
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
            with open('data/shown_history.json', 'w') as file_handle:
                json.dump(serializable, file_handle, indent=2)
        except Exception as e:
            print(f"Error saving shown history: {e}")

    def _load_transition_matrix(self) -> Dict:
        if os.path.exists(self.transition_path):
            try:
                with open(self.transition_path, 'r') as file_handle:
                    loaded = json.load(file_handle)
                    return loaded if isinstance(loaded, dict) else {}
            except Exception as e:
                print(f"Error loading transitions: {e}")
        return {}

    def _save_transition_matrix(self):
        try:
            os.makedirs('data', exist_ok=True)
            with open(self.transition_path, 'w') as file_handle:
                json.dump(self.transition_matrix, file_handle, indent=2)
        except Exception as e:
            print(f"Error saving transitions: {e}")

    def _append_reward_history(self, entry: Dict):
        self.reward_history.append(entry)
        self._save_json_list(self.reward_history_path, self.reward_history)

    def _append_outcome(self, entry: Dict):
        self.recommendation_outcomes.append(entry)
        self._save_json_list(self.outcome_path, self.recommendation_outcomes)

    # ---------------------------
    # Core RL Functions
    # ---------------------------

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

        if key not in self.q_table or not items:
            self.initialize_emotion(emotion, items, content_type)

        shown = self.shown_history.get(key, set())
        all_shown = len(shown) >= len(items)

        if all_shown:
            self.reset_shown(emotion, content_type)
            shown = set()

        exclude = set(exclude_indices) | shown
        valid_items = [(i, item) for i, item in enumerate(items) if i not in exclude]

        if not valid_items:
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
                    key=lambda x: self.q_table[key].get(str(x[1].get(id_field, hash(str(x[1])))), 0.0)
                )

        selected_index = choice[0]
        self.mark_shown(emotion, content_type, selected_index)
        return selected_index

    def _normalize_reward(self, reward: float) -> float:
        return max(-1.0, min(1.0, float(reward)))

    def _transition_score(self, previous_emotion: Optional[str], next_emotion: Optional[str]) -> float:
        if not previous_emotion or not next_emotion:
            return 0.0

        previous_emotion = previous_emotion.lower()
        next_emotion = next_emotion.lower()

        if previous_emotion in self.transition_scores and next_emotion in self.transition_scores[previous_emotion]:
            return self.transition_scores[previous_emotion][next_emotion]

        valence = {
            'happy': 1.0,
            'relaxed': 0.7,
            'calm': 0.6,
            'neutral': 0.2,
            'surprise': 0.1,
            'sad': -0.6,
            'fear': -0.7,
            'disgust': -0.75,
            'angry': -0.9,
            'stressed': -0.8,
        }
        return max(-1.0, min(1.0, valence.get(next_emotion, 0.0) - valence.get(previous_emotion, 0.0)))

    def calculate_reward(
        self,
        feedback_score: float = 0.0,
        engagement_score: float = 0.0,
        mood_improvement_score: float = 0.0,
        completion_score: float = 0.0,
    ) -> float:
        raw_reward = (
            self.reward_weights['feedback_score'] * feedback_score +
            self.reward_weights['engagement_score'] * engagement_score +
            self.reward_weights['mood_improvement_score'] * mood_improvement_score +
            self.reward_weights['completion_score'] * completion_score
        )
        return self._normalize_reward(raw_reward)

    def record_transition(
        self,
        previous_emotion: str,
        next_emotion: str,
        content_type: str,
        item_id: str,
        reward: float = 0.0,
    ):
        previous_emotion = previous_emotion.lower()
        next_emotion = next_emotion.lower()
        content_type = content_type.lower()

        if previous_emotion not in self.transition_matrix:
            self.transition_matrix[previous_emotion] = {}
        if next_emotion not in self.transition_matrix[previous_emotion]:
            self.transition_matrix[previous_emotion][next_emotion] = {'count': 0, 'by_content': {}}

        self.transition_matrix[previous_emotion][next_emotion]['count'] += 1
        by_content = self.transition_matrix[previous_emotion][next_emotion]['by_content']
        by_content[content_type] = by_content.get(content_type, 0) + 1

        transition_score = self._transition_score(previous_emotion, next_emotion)
        self.transition_history.append({
            'timestamp': datetime.now().isoformat(),
            'previous_emotion': previous_emotion,
            'next_emotion': next_emotion,
            'content_type': content_type,
            'item_id': str(item_id),
            'reward': float(reward),
            'transition_score': float(transition_score),
        })

        self._save_transition_matrix()

    def update_q_value(
        self,
        emotion: str,
        item_id: str,
        reward: float = None,
        content_type: str = 'song',
        feedback_score: float = None,
        engagement_score: float = 0.0,
        mood_improvement_score: float = 0.0,
        completion_score: float = 0.0,
        previous_emotion: str = None,
        next_emotion: str = None,
        metadata: Dict = None,
    ):
        """Update Q-value using a normalized multi-factor reward."""
        start_time = time.time()

        emotion = emotion.lower()
        content_type = content_type.lower()
        key = (emotion, content_type)

        if metadata:
            feedback_score = metadata.get('feedback_score', feedback_score)
            engagement_score = metadata.get('engagement_score', engagement_score)
            mood_improvement_score = metadata.get('mood_improvement_score', mood_improvement_score)
            completion_score = metadata.get('completion_score', completion_score)
            previous_emotion = metadata.get('previous_emotion', previous_emotion)
            next_emotion = metadata.get('next_emotion', next_emotion)

        if feedback_score is None:
            feedback_score = reward if reward is not None else 0.0

        if not mood_improvement_score and previous_emotion and next_emotion:
            mood_improvement_score = self._transition_score(previous_emotion, next_emotion)

        normalized_reward = self.calculate_reward(
            feedback_score=feedback_score,
            engagement_score=engagement_score,
            mood_improvement_score=mood_improvement_score,
            completion_score=completion_score,
        )

        if key not in self.q_table:
            self.q_table[key] = {}

        item_id = str(item_id)
        if item_id not in self.q_table[key]:
            self.q_table[key][item_id] = 0.0

        current_q = self.q_table[key][item_id]
        max_future_q = max(self.q_table[key].values()) if self.q_table[key] else 0.0
        new_q = current_q + self.alpha * (normalized_reward + self.gamma * max_future_q - current_q)
        self.q_table[key][item_id] = new_q
        self.convergence_history.append(abs(new_q - current_q))

        self.stats['total_interactions'] += 1
        if normalized_reward > 0:
            self.stats['likes'] += 1
        else:
            self.stats['dislikes'] += 1
        self.stats['emotion_interactions'][emotion] = self.stats['emotion_interactions'].get(emotion, 0) + 1
        if content_type not in self.stats['content_stats']:
            self.stats['content_stats'][content_type] = {}
        self.stats['content_stats'][content_type][emotion] = self.stats['content_stats'][content_type].get(emotion, 0) + 1

        if previous_emotion and next_emotion:
            self.record_transition(previous_emotion, next_emotion, content_type, item_id, normalized_reward)

        reward_entry = {
            'timestamp': datetime.now().isoformat(),
            'emotion': emotion,
            'item_id': item_id,
            'content_type': content_type,
            'feedback_score': float(feedback_score),
            'engagement_score': float(engagement_score),
            'mood_improvement_score': float(mood_improvement_score),
            'completion_score': float(completion_score),
            'normalized_reward': float(normalized_reward),
            'previous_emotion': previous_emotion,
            'next_emotion': next_emotion,
            'q_before': float(current_q),
            'q_after': float(new_q),
        }
        self._append_reward_history(reward_entry)
        self._append_outcome({
            'timestamp': reward_entry['timestamp'],
            'emotion': emotion,
            'content_type': content_type,
            'item_id': item_id,
            'success': normalized_reward > 0,
            'normalized_reward': float(normalized_reward),
        })

        log_event('reward_update', reward_entry)

        self._save_q_table()
        self._save_stats()

        elapsed = time.time() - start_time
        self.rl_training_time += elapsed

        total = self.stats['total_interactions']
        avg_time = self.rl_training_time / total if total > 0 else 0
        print(f"[RL] Total training computation time: {self.rl_training_time:.6f}s | Average per update: {avg_time:.6f}s")

        return normalized_reward

    # ---------------------------
    # Analytics
    # ---------------------------

    def _rolling_average(self, values: List[float], window: int = 5) -> List[float]:
        if not values:
            return []
        trend = []
        for index in range(len(values)):
            start_index = max(0, index - window + 1)
            window_values = values[start_index:index + 1]
            trend.append(sum(window_values) / len(window_values))
        return trend

    def get_analytics(self) -> Dict:
        rewards = [float(entry.get('normalized_reward', 0.0)) for entry in self.reward_history]
        reward_trend = self._rolling_average(rewards, window=5)
        average_reward = sum(rewards) / len(rewards) if rewards else 0.0
        success_rate = (sum(1 for reward in rewards if reward > 0) / len(rewards) * 100) if rewards else 0.0
        emotional_improvement = (
            sum(1 for transition in self.transition_history if transition.get('transition_score', 0.0) > 0)
            / len(self.transition_history) * 100
        ) if self.transition_history else 0.0
        recent_delta = list(self.convergence_history)[-20:]
        convergence_score = max(0.0, 1.0 - (sum(recent_delta) / len(recent_delta))) if recent_delta else 0.0

        transition_summary = {}
        for previous_emotion, next_states in self.transition_matrix.items():
            transition_summary[previous_emotion] = {}
            for next_emotion, payload in next_states.items():
                transition_summary[previous_emotion][next_emotion] = payload.get('count', 0)

        emotion_distribution = {}
        for entry in self.reward_history:
            emotion_name = entry.get('emotion', 'neutral')
            emotion_distribution[emotion_name] = emotion_distribution.get(emotion_name, 0) + 1

        return {
            'average_reward': round(average_reward, 3),
            'success_rate': round(success_rate, 1),
            'emotional_improvement_percentage': round(emotional_improvement, 1),
            'convergence_score': round(convergence_score, 3),
            'reward_trend': [round(value, 3) for value in reward_trend[-25:]],
            'reward_history': self.reward_history[-25:],
            'transition_summary': transition_summary,
            'emotion_distribution': emotion_distribution,
            'transition_events': list(self.transition_history),
        }

    def get_stats(self) -> Dict:
        total = self.stats['total_interactions']
        like_rate = (self.stats['likes'] / total * 100) if total > 0 else 0
        analytics = self.get_analytics()
        return {
            'total_interactions': total,
            'like_rate': round(like_rate, 1),
            'likes': self.stats['likes'],
            'dislikes': self.stats['dislikes'],
            'average_reward': analytics['average_reward'],
            'success_rate': analytics['success_rate'],
            'emotional_improvement_percentage': analytics['emotional_improvement_percentage'],
            'convergence_score': analytics['convergence_score'],
            'reward_trend': analytics['reward_trend'],
        }

    def get_rl_training_time(self) -> float:
        return self.rl_training_time

    def reset_rl_training_time(self):
        self.rl_training_time = 0.0
