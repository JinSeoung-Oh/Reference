# From https://medium.com/towards-artificial-intelligence/reinforcement-learning-multi-agent-cooperation-with-madqn-part-5-96456e8c32e2

# The DQN network architecture
class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.output_dim = output_dim
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )

    def forward(self, x):
        return self.fc(x)

# The Replay Buffer
class ReplayBuffer:
    def __init__(self, buffer_size):
        self.buffer = deque(maxlen=buffer_size)
        self.rewards = deque(maxlen=buffer_size)  

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
        self.rewards.append(float(reward))  

    def sample(self, batch_size):
        states, actions, rewards, next_states, dones = zip(*random.sample(self.buffer, batch_size))
        return np.array(states), np.array(actions), np.array(rewards), np.array(next_states), np.array(dones)

    def __len__(self):
        return len(self.buffer)


    
class DQNAgent:
    def __init__(self, input_dim, output_dim, gamma=0.99, lr=1e-3, tau=0.002):
        self.policy_network = DQN(input_dim, output_dim).float()
        self.target_network = DQN(input_dim, output_dim).float()
        self.target_network.load_state_dict(self.policy_network.state_dict())
        self.optimizer = optim.Adam(self.policy_network.parameters(), lr=lr)
        self.gamma = gamma
        self.tau = tau
        self.replay_buffer = ReplayBuffer(125000)

    def act(self, state, epsilon=0.1):
        if np.random.rand() < epsilon:
            return np.random.randint(env.action_space[0].n)
        state = torch.FloatTensor(state).unsqueeze(0)
        q_values = self.policy_network(state)
        return q_values.argmax().item()

    def soft_update(self):
        for target_param, policy_param in zip(self.target_network.parameters(), self.policy_network.parameters()):
            target_param.data.copy_(self.tau * policy_param.data + (1 - self.tau) * target_param.data)

    def train(self, batch_size=128):

        state, action, reward, next_state, done = self.replay_buffer.sample(batch_size)
        state = torch.FloatTensor(state)
        action = torch.LongTensor(action)
        reward = torch.FloatTensor(reward)
        next_state = torch.FloatTensor(next_state)
        done = torch.FloatTensor(done)

        q_values = self.policy_network(state)
        next_q_values = self.target_network(next_state)
        q_value = q_values.gather(1, action.unsqueeze(1)).squeeze(1)
        next_q_value = next_q_values.max(1)[0]
        q_target = reward + self.gamma * next_q_value * (1 - done)

        loss = nn.functional.mse_loss(q_value, q_target.detach())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.soft_update()

  # Initialize the gym environment
env = gym.make('ma_gym:Switch4-v0', max_steps=250)
num_agents = env.n_agents
input_dim = env.observation_space[0].shape[0]
output_dim = env.action_space[0].n

# Initialize the agents
agent0 = DQNAgent(input_dim*num_agents, output_dim)
agent1 = DQNAgent(input_dim*num_agents, output_dim)
agent2 = DQNAgent(input_dim*num_agents, output_dim)
agent3 = DQNAgent(input_dim*num_agents, output_dim)

num_episodes = 3000
batch_size = 128
epsilon_start = 1.0
epsilon_end = 0.01
epsilon_decay = 0.99
epsilon = epsilon_start

moving_rewards = deque(maxlen=10)

for episode in range(num_episodes):
    done_n = [False for _ in range(env.n_agents)]
    state_n = env.reset()
    episode_reward = 0

    while not all(done_n):
        
        actions_n = []
        full_state = np.concatenate(state_n)

        for agent in [agent0, agent1, agent2, agent3]:
            action = agent.act(full_state, epsilon)
            actions_n.append(action)

        next_state_n, reward_n, done_n, _ = env.step(actions_n)
        full_next_state = np.concatenate(next_state_n)
        
        # Store the experience in replay buffer
        agent0.replay_buffer.push(full_state, actions_n[0], sum(reward_n), full_next_state,  all(done_n))
        agent1.replay_buffer.push(full_state, actions_n[1], sum(reward_n), full_next_state,  all(done_n))
        agent2.replay_buffer.push(full_state, actions_n[2], sum(reward_n), full_next_state,  all(done_n))
        agent3.replay_buffer.push(full_state, actions_n[3], sum(reward_n), full_next_state,  all(done_n))

        state_n = next_state_n
        episode_reward += sum(reward_n)

        # Update the networks
        if len(agent0.replay_buffer) > batch_size:
            agent0.train(batch_size)
        if len(agent1.replay_buffer) > batch_size:
            agent1.train(batch_size)
        if len(agent2.replay_buffer) > batch_size:
            agent2.train(batch_size)
        if len(agent3.replay_buffer) > batch_size:
            agent3.train(batch_size)

    epsilon = max(epsilon_end, epsilon * epsilon_decay)

    moving_rewards.append(episode_reward)
    avg_reward = sum(moving_rewards) / len(moving_rewards)

    if (episode + 1) % 10 == 0:
        print(f"Episode {episode+1}, Average Reward over last 10 episodes: {avg_reward:.2f}")

    if avg_reward >= 14 and episode_reward >= 16:
        print(f"Training completed successfully at Episode {episode+1}, with average reward over 10 episodes: {avg_reward:.2f}")
        break

num_episodes = 20
delay = 0.1 

for _ in range(num_episodes):
    done_n = [False for _ in range(env.n_agents)]
    ep_reward = 0

    state_n = env.reset()
    while not all(done_n):
        env.render()
        actions_n = []
        all_states = np.concatenate(state_n)
        time.sleep(delay)  
        for agent in [agent0, agent1,agent2,agent3]:
            action = agent.act(all_states, epsilon=0)
            actions_n.append(action)
        state_n, reward_n, done_n, info = env.step(actions_n)
        ep_reward += sum(reward_n)

env.close()

## Implementing CTDE MADQN
# Initialize the gym environment
env = gym.make('ma_gym:Switch4-v0', max_steps=250)
num_agents = env.n_agents
input_dim = env.observation_space[0].shape[0]
output_dim = env.action_space[0].n

# Initialize the agents
agent = DQNAgent(input_dim*4+1, output_dim)

num_episodes = 3000
batch_size = 128
epsilon_start = 1.0
epsilon_end = 0.1
epsilon_decay = 0.99
epsilon = epsilon_start

moving_rewards = deque(maxlen=10)

for episode in range(num_episodes):
    done_n = [False for _ in range(env.n_agents)]
    state_n = env.reset()
    episode_reward = 0

    while not all(done_n):
        
        full_state_agent_0 = np.concatenate([np.concatenate(state_n),np.array([0])]) 
        full_state_agent_1 = np.concatenate([np.concatenate(state_n),np.array([1])]) 
        full_state_agent_2 = np.concatenate([np.concatenate(state_n),np.array([2])])
        full_state_agent_3 = np.concatenate([np.concatenate(state_n),np.array([3])])

        action_agent_0 = agent.act(full_state_agent_0, epsilon)  
        action_agent_1 = agent.act(full_state_agent_1, epsilon)
        action_agent_2 = agent.act(full_state_agent_2, epsilon) 
        action_agent_3 = agent.act(full_state_agent_3, epsilon)

        all_actions = [action_agent_0,action_agent_1,action_agent_2,action_agent_3]

        next_state_n, reward_n, done_n, _ = env.step(all_actions)

        full_next_state_agent_0 = np.concatenate([np.concatenate(next_state_n),np.array([0])]) 
        full_next_state_agent_1 = np.concatenate([np.concatenate(next_state_n),np.array([1])]) 
        full_next_state_agent_2 = np.concatenate([np.concatenate(next_state_n),np.array([2])])
        full_next_state_agent_3 = np.concatenate([np.concatenate(next_state_n),np.array([3])])

        full_rewards = sum(reward_n)
        
        # Store the experience in replay buffer
        agent.replay_buffer.push(full_state_agent_0, action_agent_0, reward_n[0], full_next_state_agent_0, all(done_n))
        agent.replay_buffer.push(full_state_agent_1, action_agent_1, reward_n[1], full_next_state_agent_1, all(done_n))
        agent.replay_buffer.push(full_state_agent_2, action_agent_2, reward_n[2], full_next_state_agent_2, all(done_n))
        agent.replay_buffer.push(full_state_agent_3, action_agent_3, reward_n[3], full_next_state_agent_3, all(done_n))

        state_n = next_state_n
        episode_reward += sum(reward_n)

        # Update the networks
        if len(agent.replay_buffer) > batch_size:
            agent.train(batch_size)

    epsilon = max(epsilon_end, epsilon * epsilon_decay)

    moving_rewards.append(episode_reward)
    avg_reward = sum(moving_rewards) / len(moving_rewards)

    if (episode + 1) % 10 == 0:
        print(f"Episode {episode+1}, Average Reward over last 10 episodes: {avg_reward:.2f}")

    if avg_reward >= 14 and episode_reward>=16:
        print(f"Training completed successfully at Episode {episode+1}, with average reward over 10 episodes: {avg_reward:.2f}")
        break

##### Change the direction of each agent
env = gym.make('ma_gym:Switch4-v0', max_steps=100)
num_episodes = 20
delay = 0.1 

for _ in range(num_episodes):
    done_n = [False for _ in range(env.n_agents)]
    ep_reward = 0

    state_n = env.reset()
    while not all(done_n):
        env.render()
        time.sleep(delay)  
        full_state_agent_0 = np.concatenate([np.concatenate(state_n),np.array([0])]) 
        full_state_agent_1 = np.concatenate([np.concatenate(state_n),np.array([1])]) 
        full_state_agent_2 = np.concatenate([np.concatenate(state_n),np.array([2])])
        full_state_agent_3 = np.concatenate([np.concatenate(state_n),np.array([3])])

        action_agent_0 = agent.act(full_state_agent_0, epsilon=0)  
        action_agent_1 = agent.act(full_state_agent_1, epsilon=0)
        action_agent_2 = agent.act(full_state_agent_2, epsilon=0) 
        action_agent_3 = agent.act(full_state_agent_3, epsilon=0)

        all_actions = [action_agent_0,action_agent_1,action_agent_2,action_agent_3]

        state_n, reward_n, done_n, _ = env.step(all_actions)
        ep_reward += sum(reward_n)

env.close()

## Implementing CTCE MADQN
# The DQN network architecture
class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )

    def forward(self, x):
        return self.fc(x)

# The Replay Buffer
class ReplayBuffer:
    def __init__(self, buffer_size):
        self.buffer = deque(maxlen=buffer_size)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        states, actions, rewards, next_states, dones = zip(*random.sample(self.buffer, batch_size))
        return np.array(states), np.array(actions), np.array(rewards), np.array(next_states), np.array(dones)

    def __len__(self):
        return len(self.buffer)

    
class DQNAgent:
    def __init__(self, input_dim, output_dim, gamma=0.99, lr=1e-3, tau=0.002):
        self.input_dim = input_dim 
        self.output_dim = output_dim
        self.policy_network = DQN(self.input_dim, self.output_dim).float()  
        self.target_network = DQN(self.input_dim, self.output_dim).float()
        self.target_network.load_state_dict(self.policy_network.state_dict())
        self.optimizer = optim.Adam(self.policy_network.parameters(), lr=lr)
        self.gamma = gamma
        self.tau = tau
        self.replay_buffer = ReplayBuffer(125000)

    def act(self, full_state, epsilon=0.1):
        if np.random.rand() < epsilon:
            return [np.random.randint(env.action_space[0].n) for _ in range(4)]
        full_state = torch.FloatTensor(full_state).unsqueeze(0)
        q_values = self.policy_network(full_state)
        q_values = q_values.view(4, -1)
        return q_values.argmax(dim=1).tolist()

    def soft_update(self):
        for target_param, policy_param in zip(self.target_network.parameters(), self.policy_network.parameters()):
            target_param.data.copy_(self.tau * policy_param.data + (1 - self.tau) * target_param.data)

    def train(self, batch_size=128):
        state, action, reward, next_state, done = self.replay_buffer.sample(batch_size)
        state = torch.FloatTensor(state)
        next_state = torch.FloatTensor(next_state)
        reward = torch.FloatTensor(reward).unsqueeze(-1)  
        done = torch.FloatTensor(done).unsqueeze(-1)  
        action = torch.LongTensor(action)

        q_values = self.policy_network(state).view(batch_size, 4, -1)
        next_q_values = self.target_network(next_state).view(batch_size, 4, -1)

        q_value = q_values.gather(-1, action.view(batch_size, 4, 1)).squeeze(-1)
        next_q_value, _ = next_q_values.max(-1)
        
        q_target = reward + (self.gamma * next_q_value * (1 - done))
        loss = nn.functional.mse_loss(q_value, q_target.detach())

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.soft_update()

  # Initialize the gym environment
env = gym.make('ma_gym:Switch4-v0', max_steps=250)
num_agents = env.n_agents
input_dim = env.observation_space[0].shape[0]
output_dim = env.action_space[0].n

# Initialize the agents
central_agent = DQNAgent(input_dim*4, output_dim*4)

num_episodes = 3000
batch_size = 64
epsilon_start = 1.0
epsilon_end = 0.01
epsilon_decay = 0.99
epsilon = epsilon_start

moving_rewards = deque(maxlen=10)

for episode in range(num_episodes):
    done_n = [False for _ in range(env.n_agents)]
    state_n = env.reset()
    episode_reward = 0

    while not all(done_n):
        
        full_state = np.concatenate(state_n)  
        actions_n = central_agent.act(full_state, epsilon)  

        next_state_n, reward_n, done_n, _ = env.step(actions_n)
        full_next_state = np.concatenate(next_state_n)
        full_rewards = sum(reward_n)
        
        # Store the experience in replay buffer
        central_agent.replay_buffer.push(full_state, actions_n, full_rewards, full_next_state, all(done_n))

        state_n = next_state_n
        episode_reward += sum(reward_n)

        # Update the networks
        if len(central_agent.replay_buffer) > batch_size:
            central_agent.train(batch_size)

    epsilon = max(epsilon_end, epsilon * epsilon_decay)

    moving_rewards.append(episode_reward)
    avg_reward = sum(moving_rewards) / len(moving_rewards)

    if (episode + 1) % 10 == 0:
        print(f"Episode {episode+1}, Average Reward over last 10 episodes: {avg_reward:.2f}")

    if avg_reward >= 14 and episode_reward>=16:
        print(f"Training completed successfully at Episode {episode+1}, with average reward over 10 episodes: {avg_reward:.2f}")
        break

#####
env = gym.make('ma_gym:Switch4-v0', max_steps=100)
num_episodes = 20
delay = 0.1 

for _ in range(num_episodes):
    done_n = [False for _ in range(env.n_agents)]
    ep_reward = 0
    state_n = env.reset()
    while not all(done_n):
        env.render()
        time.sleep(delay)  
        full_state = np.concatenate(state_n)  
        actions_n = central_agent.act(full_state, epsilon=0)  
        state_n, reward_n, done_n, info = env.step(actions_n)
        ep_reward += sum(reward_n)

env.close()
