import gym
import numpy as np
from collections import defaultdict

# Epsilon-greedy policy for exploration and exploitation
def epsilon_greedy_policy(Q, state, nA, epsilon):
    if np.random.rand() < epsilon:
        return np.random.choice(nA)  # Explore (random action)
    else:
        return np.argmax(Q[state])   # Exploit (best action)

# Generate an episode using the epsilon-greedy policy
def generate_episode(env, Q, epsilon):
    episode = []
    state = env.reset()  # Reset environment to start new episode
    done = False

    while not done:
        action = epsilon_greedy_policy(Q, state, env.action_space.n, epsilon)
        next_state, reward, done, _ = env.step(action)
        episode.append((state, action, reward))  # Store (state, action, reward)
        state = next_state

    return episode

# Update Q-values based on the episode generated
def update_Q(Q, episode, alpha, gamma):
    G = 0  # Initialize return
    for state, action, reward in reversed(episode):  # Process episode backward
        G = gamma * G + reward  # Accumulate return
        # Update Q-value for the state-action pair using the observed return G
        Q[state][action] += alpha * (G - Q[state][action])

# Training function to run Monte Carlo simulation over multiple episodes
def train(env, num_episodes, alpha=0.1, gamma=1.0, epsilon=0.1):
    Q = defaultdict(lambda: np.zeros(env.action_space.n))  # Initialize Q-table
    for episode in range(num_episodes):
        # Generate an episode
        episode_data = generate_episode(env, Q, epsilon)
        # Update Q-values using the episode
        update_Q(Q, episode_data, alpha, gamma)
    return Q

# Function to test the policy after training
def test_policy(Q, env, num_episodes):
    wins, losses, draws = 0, 0, 0
    for _ in range(num_episodes):
        state = env.reset()
        done = False
        while not done:
            action = np.argmax(Q[state])  # Always choose the best action
            state, reward, done, _ = env.step(action)
        if reward > 0:
            wins += 1
        elif reward < 0:
            losses += 1
        else:
            draws += 1
    print(f'Wins: {wins}, Losses: {losses}, Draws: {draws}')

# Main code to run the training and testing
if __name__ == '__main__':
    env = gym.make('Blackjack-v1')  # Load Blackjack environment from gym

    # Hyperparameters
    num_training_episodes = 500000  # Number of episodes for training
    alpha = 0.1  # Learning rate
    gamma = 1.0  # Discount factor
    epsilon = 0.1  # Epsilon for epsilon-greedy policy

    # Train the Q-learning agent
    Q = train(env, num_training_episodes, alpha, gamma, epsilon)

    # Test the learned policy after training
    num_testing_episodes = 1000
    test_policy(Q, env, num_testing_episodes)
