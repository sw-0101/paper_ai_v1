import numpy as np
import random

class PaperRecommendationEnv:
    def __init__(self, all_paper_titles, embedding_function):
        self.all_paper_titles = all_paper_titles
        self.embedding_function = embedding_function
        self.num_recommended_papers = 5
        self.state = None

    def reset(self):
        self.state = random.sample(self.all_paper_titles, 10)
        return [self.embedding_function(title) for title in self.state]

    def get_user_feedback(self, recommended_papers):
        feedback = [random.choice(["like", "dislike"]) for _ in recommended_papers]
        return feedback

    def step(self, action):
        recommended_papers = [self.state[i] for i in action]
        feedback = self.get_user_feedback(recommended_papers)
        reward = sum([1 if fb == "like" else -1 for fb in feedback])
        
        done = True
        next_state = self.reset()
        return next_state, reward, done