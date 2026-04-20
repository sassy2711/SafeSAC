import os
import re
import json
import numpy as np
from openai import OpenAI
import ast

class SafeLLMExplorerContinuous:
    def __init__(self, api_key=None, model="gpt-4o-mini", temperature=0.7):
        """
        Initializes the Safe LLM Explorer.
        Temperature is set to 0.7 to prioritize logical rule adherence over highly random exploration.
        """
        self.model = model
        self.temperature = temperature 
        self.client = OpenAI(api_key=api_key)

    def _format_episode_data(self, action_sequences, rewards, violations, m_samples=50):
        """
        Downsamples the action sequences and pairs them with their respective 
        rewards and violation counts so they fit cleanly in the LLM context window.
        """
        formatted_data = []
        for i, (actions, rew, viol) in enumerate(zip(action_sequences, rewards, violations)):
            # Downsample the action sequence to M actions to fit context window limits
            if len(actions) > m_samples:
                idx = np.linspace(0, len(actions) - 1, m_samples, dtype=int)
                sampled_actions = [actions[j] for j in idx]
            else:
                sampled_actions = actions
            
            # Round continuous actions to 2 decimal places to save tokens
            rounded_actions = [np.round(act, 2).tolist() for act in sampled_actions]
            
            formatted_data.append(
                f"Episode {i+1}:\n"
                f"  - Total Reward: {rew:.2f}\n"
                f"  - Safety Violations: {viol}\n"
                f"  - Action Sequence: {rounded_actions}\n"
            )
            
        return "\n".join(formatted_data)

    def stage_1_safety_analysis(self, formatted_data, task_description, verbal_constraints):
        """
        Stage 1: Analyzes which action patterns led to safety violations vs. rewards.
        """
        print("Calling LLM Stage 1 (Safe Trajectory Analysis)...")

        prompt1 = f"""
You are an expert Safe Reinforcement Learning analyst. Your goal is to help an agent optimize safe exploration.

{task_description}

CRITICAL SAFETY CONSTRAINTS:
{verbal_constraints}

Below is the data from the agent's recent episodes. For each episode, you will see the total reward, the number of safety violations triggered, and a sampled sequence of the continuous actions taken.
(A higher number of violations means the agent's actions heavily breached the constraints).

{formatted_data}

Please analyze the relationship between the action sequences, the rewards, and the number of violations. 
Generate a description of what the agent is doing wrong, and provide a strategy recommendation that balances task progression (maximizing reward) with strict adherence to the safety constraints (minimizing violations).
"""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a Safe RL analyst prioritizing zero-violation policies."},
                    {"role": "user", "content": prompt1}
                ],
                temperature=self.temperature
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"[LLM Stage 1 Error]: {e}")
            return ""

    def stage_2_get_safe_bias(self, summary_text, task_description, verbal_constraints, action_dim):
        """
        Stage 2: Generates a directional bias to steer the agent away from violations.
        """
        print("Calling LLM Stage 2 (Safe Gaussian Bias Generation)...")
        
        output_format = f"""
Output Format (Continuous Action): 
The approach is to add a Gaussian noise to each dimension of action, and you need to decide the bias of the Gaussian noise for each dimension to encourage safe exploration.
Please output the bias for each of the {action_dim} dimensions of actions for the next episode in decimal form.
Your output format should be strictly:
{{1: [bias], 2: [bias]}}
"""

        prompt2 = f"""
You are determining the probability distribution for safe action exploration in reinforcement learning.

{task_description}

CRITICAL SAFETY CONSTRAINTS:
{verbal_constraints}

Here is the safety analysis of the agent's recent behavior:
{summary_text}

Based on this analysis and the constraints, please determine the Gaussian bias vector that will steer the agent's future action exploration away from hazardous behaviors while still allowing it to explore effectively toward the goal.

{output_format}
"""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a Safe RL strategist."},
                    {"role": "user", "content": prompt2}
                ],
                temperature=self.temperature
            )

            # ... inside stage_2_get_safe_bias ...
            text = response.choices[0].message.content
            
            # 1. Clean the text: strip markdown code blocks and whitespace
            clean_text = re.sub(r'```json|```', '', text).strip()
            
            bias_vector = np.zeros(action_dim)
            match = re.search(r'\{.*?\}', clean_text, re.DOTALL)
            
            if match:
                try:
                    raw_dict_str = match.group(0)
                    # ast.literal_eval safely evaluates strings that look like dictionaries
                    bias_dict = ast.literal_eval(raw_dict_str)
                    
                    # Convert to numpy array
                    bias_vector = np.array([float(bias_dict[i+1][0]) for i in range(action_dim)])
                    
                except Exception as e:
                    print(f"Parsing failed: {e}")
            
            return bias_vector, text
        except Exception as e:
            print(f"[LLM Stage 2 Error]: {e}")
            return np.zeros(action_dim), str(e)

    def get_safe_exploration_bias(self, action_sequences, episode_rewards, episode_violations, task_description, action_dim, verbal_constraints, m_samples=50):
        """
        Main interface: Takes multiple episodes of actions, rewards, and violations, 
        and returns the safe Gaussian bias vector.
        """
        # 1. Format the multi-episode data
        formatted_data = self._format_episode_data(action_sequences, episode_rewards, episode_violations, m_samples)
        
        # 2. Get Analysis
        summary = self.stage_1_safety_analysis(formatted_data, task_description, verbal_constraints)
        
        if not summary:
            return np.zeros(action_dim), "Stage 1 Failed", "Stage 1 Failed"
            
        # 3. Get Bias
        bias_vector, raw_output = self.stage_2_get_safe_bias(summary, task_description, verbal_constraints, action_dim)
        
        return bias_vector, summary, raw_output


# ==========================================================
# EXAMPLE USAGE
# ==========================================================
if __name__ == "__main__":
    
    # 1. Define the Task and Safety Constraints
    TASK_DESC = (
        "The agent is navigating a 2D continuous space from [0.0, 0.0] to a goal at [4.0, 4.0]. "
        "The action space consists of 2 continuous values between -1.0 and 1.0 representing movement vectors (X, Y)."
    )
    
    CONSTRAINTS = (
        "The agent MUST NOT enter the rectangular hazard zone defined by the box from (1.0, 1.0) to (3.0, 3.0). "
        "Any step inside this box is a safety violation. The agent must navigate around it to reach the goal safely."
    )
    
    # 2. Simulate some previous episodes (Actions, Rewards, Violations)
    
    # Episode 1: Moves diagonally directly into the box. Fast but unsafe.
    actions_1 = [np.array([0.5, 0.5]) for _ in range(15)] 
    reward_1 = -5.0 
    viol_1 = 8 
    
    # Episode 2: Tries to move right, skims the bottom edge of the box. Moderately safe.
    actions_2 = [np.array([0.8, 0.1]) for _ in range(15)]
    reward_2 = -8.0
    viol_2 = 2
    
    # Episode 3: Moves purely up, avoiding the box entirely initially. Slow but safe.
    actions_3 = [np.array([0.0, 1.0]) for _ in range(15)]
    reward_3 = -12.0
    viol_3 = 0
    
    # Pack them into lists
    action_sequences = [actions_1, actions_2, actions_3]
    episode_rewards = [reward_1, reward_2, reward_3]
    episode_violations = [viol_1, viol_2, viol_3]
    
    import os
    api_key = os.getenv("OPENAI_API_KEY")
    # 3. Initialize and run Safe LLM-Explorer
    explorer = SafeLLMExplorerContinuous(
        api_key=api_key,
        model="gpt-4o-mini",
        temperature=0.7 
    )
    
    print("\nRequesting Safe Exploration Bias from LLM...")
    action_bias, stage1_summary, stage2_raw = explorer.get_safe_exploration_bias(
        action_sequences=action_sequences,
        episode_rewards=episode_rewards,
        episode_violations=episode_violations,
        task_description=TASK_DESC,
        action_dim=2,
        verbal_constraints=CONSTRAINTS,
        m_samples=20 # Reduced max samples per trajectory to save context window
    )
    
    print("\n" + "="*50)
    print(f"Recommended Safe Gaussian Bias (X, Y): {action_bias}")
    print("="*50)
    print(f"\n[Stage 1 Safety Analysis]:\n{stage1_summary}")
    print(f"\n[Stage 2 Raw Output]:\n{stage2_raw}")