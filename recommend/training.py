from paper_rl_env import PaperRecommendationEnv
from paper_rl_agent import TransformerDQNAgent
from transformers import BertTokenizer, BertModel
from GoogleAPI import get_paper_titles
import torch

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')
API_KEY = "AIzaSyC230uV-vbwvHjtIxqGdXdX-0kFCrlp78o"
SEARCH_ENGINE_ID = "930c007e311a24f1c"

user_input = "segmentation"
all_paper_titles = get_paper_titles(user_input, API_KEY, SEARCH_ENGINE_ID)

def get_bert_embedding(sentence):
    inputs = tokenizer(sentence, return_tensors="pt", truncation=True, padding=True, max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze()

# all_paper_titles = [
#     "Neural Machine Translation by Jointly Learning to Align and Translate",
#     "Segment Anything",
#     "Fast Segment Anything",
#     "Learning Transferable Visual Models From Natural Language Supervision",
#     "Mastering Chess and Shogi by Self-Play with a General Reinforcement Learning Algorithm",
#     "The Sequence of the Human Genome",
#     "Observation of Gravitational Waves from a Binary Black Hole Merger",
#     "Prospect Theory: An Analysis of Decision under Risk",
#     "Early Transmission Dynamics in Wuhan, China, of Novel Coronavirus–Infected Pneumonia",
#     "Trajectories of the Earth System in the Anthropocene",
# ]
#user_input = "i want a paper with segmentation"
# def get_feedback():
#     for _ in range(5):

env = PaperRecommendationEnv(all_paper_titles, user_input, get_bert_embedding)
agent = TransformerDQNAgent(input_dim=768, sequence_length=10, output_dim=5)

num_episodes = 10
epsilon = 1.0
min_epsilon = 0.1
epsilon_decay = 0.995
feedback = None
rewards = []
for episode in range(num_episodes):
    state, user_input_embedding = env.reset()
    done = False
    while not done:
        action = agent.choose_action(state, user_input_embedding, epsilon)
        #feedback = get_feedback()
        next_state, next_user_input_embedding, reward, done = env.step(action, feedback)
        agent.train(state, user_input_embedding, action, reward, next_state, next_user_input_embedding, done)
        state = next_state
        user_input_embedding = next_user_input_embedding
    epsilon = max(epsilon * epsilon_decay, min_epsilon)
    rewards.append(reward)
    print(f"Episode {episode + 1}/{num_episodes} - Reward: {reward}")

torch.save(agent.model.state_dict(), 'recommend_model.pth')
print("Training complete!")
