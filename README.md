# Agenlus Python Hub Client (`agenlus-hub`)

[![PyPI version](https://img.shields.io/pypi/v/agenlus-hub.svg)](https://pypi.org/project/agenlus-hub/)
[![Python versions](https://img.shields.io/pypi/pyversions/agenlus-hub.svg)](https://pypi.org/project/agenlus-hub/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

`agenlus-hub` is the official Python client library for the **Agenlus RL Platform**. It provides a seamless interface to download custom RL environments, run local evaluations on your PyTorch models, and automatically export, build, upload, and register them onto the Agenlus leaderboards.

---

## Key Features

- **Simple Authentication**: Quick setup with single-line token authentication.
- **Environment Downloader**: Easily fetch custom Gymnasium environment specifications and source code directly from the server.
- **Automatic Export & SB3 Support**: Exposes models (both pure PyTorch and `stable-baselines3` models) to both PyTorch (`.pt`) and ONNX (`.onnx`) formats with dynamic batching.
- **Deterministic Evaluation**: Validates model performance locally using reproducible seeds before uploading.
- **Hugging Face Hub Integration**: Automates repositories creation and stacks model uploads into tidy subfolders.
- **Leaderboard Registration**: Instant, automated registration of evaluated models to the Agenlus leaderboard.

---

## Installation

Install the package and its dependencies via `pip`:

```bash
pip install agenlus-hub
```

### Dependencies
Ensure you have the following prerequisites installed (or they will be installed automatically):
- `requests`
- `torch`
- `onnx`
- `onnxscript`
- `huggingface_hub`
- `gymnasium` (Required for local model evaluation)

---

## Quick Start

Here is a quick walkthrough showing how to log in, download an environment, build/train your model, and upload it to the leaderboard:

```python
import torch
import torch.nn as nn
import agenlus

# 1. Log in to your Agenlus account
agenlus.login(token="YOUR_AGENLUS_API_TOKEN")

# 2. Download the target environment (e.g., CartPole-v1)
# This downloads 'CartPole-v1.py' into your working directory
env_file_path = agenlus.download("system/CartPole-v1")

# 3. Define and train your PyTorch RL Model
class PolicyNet(nn.Module):
    def __init__(self, obs_dim=4, action_dim=2):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )
        
    def forward(self, x):
        return self.fc(x)

# Instantiate your model (or load from disk)
model = PolicyNet()

# 4. Evaluate and Upload to Agenlus & Hugging Face
# This evaluates the model for 100 episodes, exports to ONNX/PyTorch, 
# uploads to Hugging Face, and registers the scores on the Leaderboard.
agenlus.upload(
    model=model,
    env_id="system/CartPole-v1",
    hf_token="YOUR_HUGGINGFACE_WRITE_TOKEN",
    model_name="my-first-cartpole-agent",
    seed=42
)
```

### Stable-Baselines3 Support (DQN, PPO, SAC, etc.)

You can directly upload models trained with `stable-baselines3`. The policy network will be automatically wrapped and exported to ONNX:

```python
import gymnasium as gym
from stable_baselines3 import DQN
import agenlus

# 1. Log in
agenlus.login(token="YOUR_AGENLUS_API_TOKEN")

# 2. Download environment code (if custom or needed)
env_file = agenlus.download("system/CartPole-v1")

# 3. Define and train your SB3 agent
env = gym.make("CartPole-v1")
model = DQN("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=10000)

# 4. Upload and register to Leaderboard
# This automatically wraps model.q_net (DQN) / model.policy (PPO) for ONNX compatibility
agenlus.upload(
    model=model,
    env_id="system/CartPole-v1",
    hf_token="YOUR_HUGGINGFACE_WRITE_TOKEN",
    model_name="my-sb3-dqn-agent",
    seed=42
)
```

---

## API Reference

### `agenlus.login`
Authenticates your local client session.

```python
agenlus.login(token: str, api_url: str = None)
```
- **`token`** (str): Your Agenlus developer API token.
- **`api_url`** (str, optional): Overrides the default platform backend URL.

---

### `agenlus.download`
Downloads the source code of an environment registered on the platform.

```python
save_path = agenlus.download(env_id: str, save_path: str = None)
```
- **`env_id`** (str): The ID of the environment (e.g., `system/CartPole-v1` or `username/custom-env`).
- **`save_path`** (str, optional): Custom file path to save the code. Defaults to `{env_name}.py` in the current directory.
- **Returns**: The path to the saved file.

---

### `agenlus.upload`
Executes local evaluation, model export, Hugging Face upload, and platform leaderboard registration.

```python
agenlus.upload(
    model,
    env_id: str,
    hf_token: str,
    hf_repo: str = None,
    model_name: str = None,
    seed: int = None
)
```
- **`model`** (PyTorch model or stable-baselines3 model): The trained RL model (e.g., a PyTorch `nn.Module` or a `stable_baselines3` agent). If a stable-baselines3 model is provided, its policy network is automatically wrapped and extracted.
- **`env_id`** (str): Target environment ID on the platform.
- **`hf_token`** (str): Write access token for Hugging Face.
- **`hf_repo`** (str, optional): The Hugging Face repo (e.g., `username/repo-name`). If omitted, checks your profile settings or defaults to `username/agenlus-agents`.
- **`model_name`** (str, optional): Unique name for this model checkpoint. Subdirectories under the repository will be structured around this name.
- **`seed`** (int, optional): Seed used to ensure deterministic environment evaluation. (Default: `42`).

---

## License

This project is licensed under the MIT License - see the LICENSE file for details.
