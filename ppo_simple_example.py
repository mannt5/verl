"""
PPO 算法简化示例 - 逐步解释版本
帮助理解 PPO 的核心运行流程
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import gym


class PPOSimple:
    """简化的 PPO 实现，突出核心概念"""
    
    def __init__(self):
        # 创建神经网络
        self.policy_net = nn.Sequential(
            nn.Linear(4, 128),  # 输入层 (CartPole 有 4 个状态)
            nn.ReLU(),
            nn.Linear(128, 2)   # 输出层 (2 个动作: 左/右)
        )
        
        self.value_net = nn.Sequential(
            nn.Linear(4, 128),
            nn.ReLU(),
            nn.Linear(128, 1)   # 输出状态价值
        )
        
        # 优化器
        self.optimizer = optim.Adam(
            list(self.policy_net.parameters()) + 
            list(self.value_net.parameters()), 
            lr=0.001
        )
        
        # PPO 参数
        self.clip_epsilon = 0.2  # 裁剪范围
        self.gamma = 0.99        # 折扣因子
    
    def ppo_step_by_step(self):
        """
        PPO 算法逐步运行示例
        """
        print("PPO 算法运行步骤：\n")
        
        # ========== 步骤 1: 初始化环境 ==========
        print("【步骤 1】初始化环境")
        env = gym.make('CartPole-v1')
        state = env.reset()
        print(f"初始状态: {state}")
        print(f"状态含义: [位置, 速度, 角度, 角速度]\n")
        
        # ========== 步骤 2: 收集经验 ==========
        print("【步骤 2】使用当前策略收集经验")
        states, actions, rewards, log_probs_old, values = [], [], [], [], []
        
        for step in range(20):  # 收集 20 步经验
            # 2.1 将状态转为张量
            state_tensor = torch.FloatTensor(state)
            
            # 2.2 获取动作概率
            logits = self.policy_net(state_tensor)
            probs = torch.softmax(logits, dim=-1)
            dist = Categorical(probs)
            
            # 2.3 采样动作
            action = dist.sample()
            log_prob = dist.log_prob(action)
            
            # 2.4 获取状态价值
            value = self.value_net(state_tensor)
            
            # 2.5 执行动作
            next_state, reward, done, _ = env.step(action.item())
            
            # 2.6 存储经验
            states.append(state)
            actions.append(action.item())
            rewards.append(reward)
            log_probs_old.append(log_prob.item())
            values.append(value.item())
            
            if step < 3:  # 只打印前 3 步
                print(f"  步骤 {step+1}:")
                print(f"    动作概率: 左={probs[0]:.3f}, 右={probs[1]:.3f}")
                print(f"    选择动作: {'左' if action.item()==0 else '右'}")
                print(f"    获得奖励: {reward}")
                print(f"    状态价值: {value.item():.3f}")
            
            state = next_state
            if done:
                break
        
        print(f"  ...(共收集 {len(states)} 步经验)\n")
        
        # ========== 步骤 3: 计算优势和回报 ==========
        print("【步骤 3】计算优势函数和回报")
        
        # 3.1 计算折扣回报
        returns = []
        R = 0
        for reward in reversed(rewards):
            R = reward + self.gamma * R
            returns.insert(0, R)
        
        # 3.2 计算优势 A = R - V
        advantages = [r - v for r, v in zip(returns, values)]
        
        print(f"  示例 (前 3 步):")
        for i in range(min(3, len(returns))):
            print(f"    步骤 {i+1}: 回报={returns[i]:.3f}, "
                  f"价值={values[i]:.3f}, 优势={advantages[i]:.3f}")
        print()
        
        # ========== 步骤 4: PPO 更新 ==========
        print("【步骤 4】PPO 策略更新")
        
        # 转换为张量
        states_tensor = torch.FloatTensor(states)
        actions_tensor = torch.LongTensor(actions)
        old_log_probs_tensor = torch.FloatTensor(log_probs_old)
        returns_tensor = torch.FloatTensor(returns)
        advantages_tensor = torch.FloatTensor(advantages)
        
        # 标准化优势
        advantages_tensor = (advantages_tensor - advantages_tensor.mean()) / (advantages_tensor.std() + 1e-8)
        
        # 执行一次更新
        # 4.1 计算新的动作概率
        logits = self.policy_net(states_tensor)
        probs = torch.softmax(logits, dim=-1)
        dist = Categorical(probs)
        new_log_probs = dist.log_prob(actions_tensor)
        
        # 4.2 计算重要性采样比率
        ratio = torch.exp(new_log_probs - old_log_probs_tensor)
        
        # 4.3 计算 PPO 损失
        surr1 = ratio * advantages_tensor
        surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages_tensor
        policy_loss = -torch.min(surr1, surr2).mean()
        
        # 4.4 计算价值损失
        values_pred = self.value_net(states_tensor).squeeze()
        value_loss = nn.MSELoss()(values_pred, returns_tensor)
        
        # 4.5 总损失
        loss = policy_loss + 0.5 * value_loss
        
        print(f"  损失计算:")
        print(f"    策略损失: {policy_loss.item():.4f}")
        print(f"    价值损失: {value_loss.item():.4f}")
        print(f"    总损失: {loss.item():.4f}")
        
        # 4.6 梯度更新
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        print("\n  PPO 关键机制:")
        print(f"    - 重要性采样比率示例: {ratio[:3].detach().numpy()}")
        print(f"    - 裁剪范围: [{1-self.clip_epsilon:.1f}, {1+self.clip_epsilon:.1f}]")
        print("    - 通过裁剪防止策略更新过大，保证训练稳定性")
        
        env.close()
        print("\n【完成】PPO 一轮更新完成！")


def main():
    """运行 PPO 示例"""
    print("=" * 60)
    print("PPO (Proximal Policy Optimization) 运行示例")
    print("=" * 60)
    print("\nPPO 算法核心思想：")
    print("1. 收集一批经验数据")
    print("2. 计算优势函数评估动作好坏") 
    print("3. 使用裁剪的目标函数更新策略")
    print("4. 限制策略更新幅度，确保稳定性")
    print("\n" + "-" * 60 + "\n")
    
    # 创建并运行 PPO
    ppo = PPOSimple()
    ppo.ppo_step_by_step()
    
    print("\n" + "=" * 60)
    print("提示：这是 PPO 的简化演示，实际应用中会：")
    print("- 收集更多经验数据 (通常几千步)")
    print("- 进行多轮更新 (通常 4-10 轮)")
    print("- 使用更复杂的优势估计 (如 GAE)")
    print("- 添加熵正则化鼓励探索")
    print("=" * 60)


if __name__ == "__main__":
    main()