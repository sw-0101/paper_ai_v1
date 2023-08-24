import numpy as np
import random

class PaperRecommendationEnv:
    def __init__(self, all_paper_titles, user_input, embedding_function):
        self.all_paper_titles = all_paper_titles
        self.user_input = user_input
        self.embedding_function = embedding_function
        self.num_recommended_papers = 5
        self.state = None

    def reset(self):
        self.state = random.sample(self.all_paper_titles, 10)
        user_input_embedding = self.embedding_function(self.user_input)
        return [self.embedding_function(title) for title in self.state], user_input_embedding

    def get_user_feedback(self, recommended_papers):
        feedback = [random.choice(["like", "dislike"]) for _ in recommended_papers]
        return feedback

    def step(self, action, feedback):
        recommended_papers = [self.state[i] for i in action]
        feedback = self.get_user_feedback(recommended_papers)
        reward = sum([1 if fb == "like" else -1 for fb in feedback])
        
        done = True
        next_state, user_input_embedding = self.reset()
        return next_state, user_input_embedding, reward, done