"""
PPO (Proximal Policy Optimization) 算法示例
这个例子展示了如何使用 PPO 算法训练一个智能体来解决 CartPole 环境
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
import gym
import matplotlib.pyplot as plt
from collections import deque
import time

# 设置随机种子以确保可重复性
torch.manual_seed(42)
np.random.seed(42)


class PolicyNetwork(nn.Module):
    """
    策略网络 (Actor)
    输出每个动作的概率分布
    """
    def __init__(self, state_dim, action_dim, hidden_dim=64):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)
        
    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        action_probs = F.softmax(self.fc3(x), dim=-1)
        return action_probs


class ValueNetwork(nn.Module):
    """
    价值网络 (Critic)
    估计当前状态的价值
    """
    def __init__(self, state_dim, hidden_dim=64):
        super(ValueNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)
        
    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        value = self.fc3(x)
        return value


class PPOAgent:
    """
    PPO 智能体
    包含策略网络和价值网络，以及 PPO 算法的核心逻辑
    """
    def __init__(self, state_dim, action_dim, lr=3e-4, gamma=0.99, 
                 epsilon=0.2, c1=0.5, c2=0.01, update_epochs=10):
        # 初始化网络
        self.policy_net = PolicyNetwork(state_dim, action_dim)
        self.value_net = ValueNetwork(state_dim)
        
        # 优化器
        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.value_optimizer = optim.Adam(self.value_net.parameters(), lr=lr)
        
        # PPO 超参数
        self.gamma = gamma          # 折扣因子
        self.epsilon = epsilon      # PPO 裁剪参数
        self.c1 = c1               # 价值损失系数
        self.c2 = c2               # 熵正则化系数
        self.update_epochs = update_epochs  # 每批数据的更新轮数
        
    def select_action(self, state):
        """
        根据当前策略选择动作
        返回动作、动作概率的对数、状态价值
        """
        state = torch.FloatTensor(state).unsqueeze(0)
        
        # 获取动作概率分布
        action_probs = self.policy_net(state)
        dist = Categorical(action_probs)
        
        # 采样动作
        action = dist.sample()
        action_log_prob = dist.log_prob(action)
        
        # 获取状态价值
        state_value = self.value_net(state)
        
        return action.item(), action_log_prob, state_value
    
    def compute_returns(self, rewards, dones, next_value):
        """
        计算折扣回报 (Returns)
        使用 GAE (Generalized Advantage Estimation) 的简化版本
        """
        returns = []
        R = next_value
        
        for reward, done in zip(reversed(rewards), reversed(dones)):
            if done:
                R = 0
            R = reward + self.gamma * R
            returns.insert(0, R)
            
        return returns
    
    def compute_advantages(self, returns, values):
        """
        计算优势函数 A(s,a) = R - V(s)
        """
        advantages = []
        for ret, val in zip(returns, values):
            advantages.append(ret - val.item())
        return advantages
    
    def ppo_update(self, states, actions, old_log_probs, returns, advantages):
        """
        PPO 核心更新步骤
        """
        # 转换为张量
        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        old_log_probs = torch.FloatTensor(old_log_probs)
        returns = torch.FloatTensor(returns)
        advantages = torch.FloatTensor(advantages)
        
        # 标准化优势函数
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # 多轮更新
        for _ in range(self.update_epochs):
            # 计算新的动作概率
            action_probs = self.policy_net(states)
            dist = Categorical(action_probs)
            new_log_probs = dist.log_prob(actions)
            
            # 计算比率 r(θ) = π(a|s,θ) / π(a|s,θ_old)
            ratio = torch.exp(new_log_probs - old_log_probs)
            
            # 计算 PPO 目标的两部分
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon) * advantages
            
            # 策略损失 (取最小值实现保守更新)
            policy_loss = -torch.min(surr1, surr2).mean()
            
            # 价值损失
            values = self.value_net(states).squeeze()
            value_loss = F.mse_loss(values, returns)
            
            # 熵损失 (鼓励探索)
            entropy = dist.entropy().mean()
            
            # 总损失
            loss = policy_loss + self.c1 * value_loss - self.c2 * entropy
            
            # 更新策略网络
            self.policy_optimizer.zero_grad()
            policy_loss.backward(retain_graph=True)
            self.policy_optimizer.step()
            
            # 更新价值网络
            self.value_optimizer.zero_grad()
            value_loss.backward()
            self.value_optimizer.step()
            
        return policy_loss.item(), value_loss.item(), entropy.item()


def train_ppo():
    """
    训练 PPO 智能体
    """
    # 创建环境
    env = gym.make('CartPole-v1')
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    # 创建 PPO 智能体
    agent = PPOAgent(state_dim, action_dim)
    
    # 训练参数
    num_episodes = 500
    batch_size = 32  # 每批收集的轨迹数量
    max_steps = 200
    
    # 记录训练过程
    episode_rewards = []
    recent_rewards = deque(maxlen=100)
    
    print("开始 PPO 训练...")
    print("-" * 50)
    
    for episode in range(num_episodes):
        # 收集一批轨迹数据
        batch_states = []
        batch_actions = []
        batch_log_probs = []
        batch_rewards = []
        batch_dones = []
        batch_values = []
        
        for _ in range(batch_size):
            state = env.reset()
            episode_reward = 0
            
            states = []
            actions = []
            log_probs = []
            rewards = []
            dones = []
            values = []
            
            for step in range(max_steps):
                # 选择动作
                action, log_prob, value = agent.select_action(state)
                
                # 执行动作
                next_state, reward, done, _ = env.step(action)
                
                # 存储经验
                states.append(state)
                actions.append(action)
                log_probs.append(log_prob.item())
                rewards.append(reward)
                dones.append(done)
                values.append(value)
                
                episode_reward += reward
                state = next_state
                
                if done:
                    break
            
            # 计算最后一个状态的价值
            if not done:
                _, _, next_value = agent.select_action(next_state)
                next_value = next_value.item()
            else:
                next_value = 0
            
            # 计算回报和优势
            returns = agent.compute_returns(rewards, dones, next_value)
            advantages = agent.compute_advantages(returns, values)
            
            # 添加到批次
            batch_states.extend(states)
            batch_actions.extend(actions)
            batch_log_probs.extend(log_probs)
            batch_rewards.extend(rewards)
            batch_dones.extend(dones)
            batch_values.extend(values)
            
            # 记录奖励
            recent_rewards.append(episode_reward)
        
        # 将批次数据转换为正确的格式
        batch_returns = []
        batch_advantages = []
        
        idx = 0
        for _ in range(batch_size):
            trajectory_length = len([d for d in batch_dones[idx:] if d][0:1]) + 1
            if idx + trajectory_length > len(batch_dones):
                trajectory_length = len(batch_dones) - idx
            
            trajectory_rewards = batch_rewards[idx:idx+trajectory_length]
            trajectory_dones = batch_dones[idx:idx+trajectory_length]
            trajectory_values = batch_values[idx:idx+trajectory_length]
            
            if idx + trajectory_length < len(batch_values):
                next_value = batch_values[idx+trajectory_length].item()
            else:
                next_value = 0
                
            returns = agent.compute_returns(trajectory_rewards, trajectory_dones, next_value)
            advantages = agent.compute_advantages(returns, trajectory_values)
            
            batch_returns.extend(returns)
            batch_advantages.extend(advantages)
            
            idx += trajectory_length
            if idx >= len(batch_dones):
                break
        
        # PPO 更新
        policy_loss, value_loss, entropy = agent.ppo_update(
            batch_states[:len(batch_returns)], 
            batch_actions[:len(batch_returns)], 
            batch_log_probs[:len(batch_returns)], 
            batch_returns, 
            batch_advantages
        )
        
        # 记录平均奖励
        avg_reward = np.mean(recent_rewards)
        episode_rewards.append(avg_reward)
        
        # 打印训练信息
        if episode % 10 == 0:
            print(f"Episode {episode:4d} | "
                  f"平均奖励: {avg_reward:6.2f} | "
                  f"策略损失: {policy_loss:6.4f} | "
                  f"价值损失: {value_loss:6.4f} | "
                  f"熵: {entropy:6.4f}")
        
        # 检查是否解决了问题
        if avg_reward >= 195:
            print(f"\n问题已解决！在 {episode} 轮后达到平均奖励 {avg_reward:.2f}")
            break
    
    env.close()
    return episode_rewards, agent


def evaluate_agent(agent, num_episodes=10, render=False):
    """
    评估训练好的智能体
    """
    env = gym.make('CartPole-v1')
    
    total_rewards = []
    
    print("\n评估智能体性能...")
    print("-" * 50)
    
    for episode in range(num_episodes):
        state = env.reset()
        episode_reward = 0
        done = False
        steps = 0
        
        while not done and steps < 200:
            if render:
                env.render()
            
            # 使用训练好的策略选择动作
            action, _, _ = agent.select_action(state)
            state, reward, done, _ = env.step(action)
            episode_reward += reward
            steps += 1
        
        total_rewards.append(episode_reward)
        print(f"Episode {episode + 1}: 奖励 = {episode_reward}")
    
    env.close()
    
    print(f"\n平均奖励: {np.mean(total_rewards):.2f} ± {np.std(total_rewards):.2f}")
    return total_rewards


def plot_training_curve(episode_rewards):
    """
    绘制训练曲线
    """
    plt.figure(figsize=(10, 6))
    plt.plot(episode_rewards)
    plt.title('PPO 训练曲线 - CartPole-v1')
    plt.xlabel('训练轮数')
    plt.ylabel('平均奖励 (最近100轮)')
    plt.grid(True, alpha=0.3)
    
    # 添加解决问题的阈值线
    plt.axhline(y=195, color='r', linestyle='--', label='解决阈值 (195)')
    plt.legend()
    
    plt.savefig('/workspace/ppo_training_curve.png', dpi=300, bbox_inches='tight')
    print("\n训练曲线已保存到 ppo_training_curve.png")
    plt.close()


if __name__ == "__main__":
    print("PPO (Proximal Policy Optimization) 算法示例")
    print("=" * 50)
    print("\n环境: CartPole-v1")
    print("目标: 保持杆子直立，每个时间步获得 +1 奖励")
    print("解决标准: 连续100轮的平均奖励 >= 195")
    print("\nPPO 特点:")
    print("1. 通过裁剪确保策略更新的稳定性")
    print("2. 同时优化策略和价值函数")
    print("3. 使用多轮小批量更新提高样本效率")
    print("=" * 50)
    
    # 开始训练
    start_time = time.time()
    episode_rewards, trained_agent = train_ppo()
    training_time = time.time() - start_time
    
    print(f"\n训练完成！总用时: {training_time:.2f} 秒")
    
    # 绘制训练曲线
    plot_training_curve(episode_rewards)
    
    # 评估智能体
    evaluation_rewards = evaluate_agent(trained_agent, num_episodes=10)
    
    print("\n" + "=" * 50)
    print("PPO 算法运行完成！")