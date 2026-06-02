import os
import json
import time
import requests
import torch

CONFIG_FILE = os.path.expanduser("~/.agenlus_config.json")
DEFAULT_API_URL = "https://ai-app-store-backend.lam983039.workers.dev"

def login(token: str, api_url: str = None):
    """
    Save the user token and API URL locally.
    """
    if not api_url:
        api_url = DEFAULT_API_URL
        
    config = {
        "token": token,
        "api_url": api_url
    }
    
    with open(CONFIG_FILE, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=4)
        
    print(f"[Agenlus] Login settings saved to {CONFIG_FILE}")
    print(f"[Agenlus] API URL: {api_url}")

def upload(model: torch.nn.Module, env_id: str, hf_token: str, hf_repo: str = None, model_name: str = None, episodes: int = 100, seed: int = None):
    """
    Convert PyTorch model to ONNX + PT, run local evaluation to compute best_score,
    upload to Hugging Face (stacked under subfolder), and register to Agenlus Leaderboard.
    """
    # 1. Load config
    if not os.path.exists(CONFIG_FILE):
        raise ValueError("No login session found. Please call agenlus.login(token) first.")
        
    try:
        with open(CONFIG_FILE, "r", encoding="utf-8") as f:
            config = json.load(f)
    except Exception as e:
        raise ValueError(f"Failed to read config file: {e}")
        
    token = config.get("token")
    api_url = config.get("api_url", DEFAULT_API_URL)
    
    if not token:
        raise ValueError("Not logged in. Please run agenlus.login(token).")

    # Authenticate header for spec lookup (to retrieve profile defaults)
    headers = {
        "Authorization": f"Bearer {token}"
    }

    # 2. Retrieve env spec and dynamically resolve normalized Env ID and default HuggingFace Repo
    print(f"[Agenlus] Fetching environment specification for {env_id}...")
    normalized_env_id = env_id
    default_hf_repo = None
    try:
        spec_url = f"{api_url}/api/rl/environments/spec/{env_id}"
        response = requests.get(spec_url, headers=headers)
        if response.status_code != 200:
            raise Exception(f"Failed to fetch env spec: {response.text} (Status: {response.status_code})")
        
        env_spec_data = response.json()
        env_spec = env_spec_data.get("envSpec", {})
        obs_space = env_spec.get("observationSpace", 4)
        
        # Override with the standard normalized Env ID from the server
        normalized_env_id = env_spec_data.get("normalizedEnvId", env_id)
        default_hf_repo = env_spec_data.get("defaultHuggingFaceRepo")
        print(f"[Agenlus] Detected Env observation space: {obs_space}, Normalized Env ID: {normalized_env_id}")
    except Exception as e:
        print(f"[Warning] Failed to fetch environment spec: {e}. Falling back to default observation space size = 4.")
        obs_space = 4

    # 2.5. Local evaluation (forced evaluation to ensure model integrity)
    if seed is None:
        seed = 42
    print(f"[Agenlus] Running local evaluation on '{normalized_env_id}' for {episodes} episodes using base seed {seed}...")
    local_obs_space = None
    temp_env_path = "_temp_env.py"
    try:
        # Download the environment code to temp file
        download(normalized_env_id, temp_env_path)
        
        # Dynamically import the env
        import importlib.util
        import sys
        import inspect
        
        try:
            import gymnasium as gym
        except ImportError:
            raise ImportError("gymnasium library is required for local model evaluation. Please install it with 'pip install gymnasium'.")
            
        spec = importlib.util.spec_from_file_location("temp_env", temp_env_path)
        module = importlib.util.module_from_spec(spec)
        sys.modules["temp_env"] = module
        spec.loader.exec_module(module)
        
        # Find the Environment Class
        env_class = None
        for name, obj in inspect.getmembers(module):
            if inspect.isclass(obj):
                if obj.__module__ == "temp_env":
                    # Check if it inherits from gym.Env (checking class ancestry)
                    base_names = [base.__name__ for base in obj.__mro__]
                    if "Env" in base_names or "CartPoleEnv" in base_names or "CustomCartPole" in base_names:
                        env_class = obj
                        break
        
        if not env_class:
            # Fallback to the first class found in module that isn't imported
            classes = [obj for name, obj in inspect.getmembers(module) if inspect.isclass(obj) and obj.__module__ == "temp_env"]
            if classes:
                env_class = classes[0]
        
        if not env_class:
            raise ValueError("Could not find any environment class in the downloaded code.")
            
        print(f"[Agenlus] Found environment class: {env_class.__name__}")
        env = env_class()
        
        # Try to infer observation space from the local environment instance
        if hasattr(env, "observation_space"):
            if hasattr(env.observation_space, "shape") and env.observation_space.shape is not None:
                local_obs_space = env.observation_space.shape
            elif hasattr(env.observation_space, "n"):
                local_obs_space = env.observation_space.n
        
        # Run episodes
        total_rewards = []
        model.eval()
        
        for ep in range(episodes):
            # Flawless reproducibility: seed np_random deterministically per episode (base seed = seed)
            ep_seed = seed + ep
            try:
                reset_res = env.reset(seed=ep_seed)
            except TypeError:
                reset_res = env.reset()
                
            # Handle both gymnasium (obs, info) and old gym (obs) reset returns
            if isinstance(reset_res, tuple) and len(reset_res) == 2:
                obs, _ = reset_res
            else:
                obs = reset_res
                
            done = False
            ep_reward = 0
            while not done:
                with torch.no_grad():
                    # Convert observation to float tensor
                    # Check if observation is list/dict/numpy array
                    if isinstance(obs, dict):
                        obs_val = list(obs.values())[0] if obs else np.zeros((4,))
                    else:
                        obs_val = obs
                    
                    import numpy as np
                    obs_arr = np.array(obs_val, dtype=np.float32)
                    if local_obs_space is None:
                        local_obs_space = obs_arr.shape
                    obs_t = torch.FloatTensor(obs_arr).unsqueeze(0)
                    
                    # Forward pass
                    out = model(obs_t)
                    
                # Choose action based on Action Space
                if isinstance(env.action_space, gym.spaces.Discrete):
                    action = out.argmax(dim=-1).item()
                else:
                    action = out.squeeze(0).cpu().numpy()
                    # If action space is Box but output is flat/scalar
                    if isinstance(action, np.ndarray) and action.ndim == 0:
                        action = float(action)
                        
                step_res = env.step(action)
                # Handle both gymnasium (obs, reward, terminated, truncated, info) and old gym (obs, reward, done, info)
                if len(step_res) == 5:
                    obs, reward, terminated, truncated, _ = step_res
                    done = terminated or truncated
                else:
                    obs, reward, done, _ = step_res
                    
                ep_reward += reward
            
            total_rewards.append(ep_reward)
            
        best_score = sum(total_rewards) / len(total_rewards)
        print(f"[Agenlus] Local evaluation completed. Average Reward (bestScore) over {episodes} episodes: {best_score:.4f}")
        
    except Exception as e:
        raise Exception(f"Local evaluation failed: {e}. The model cannot be uploaded if it does not match the environment spec or fails evaluation.")
    finally:
        # Clean up temp env file
        if os.path.exists(temp_env_path):
            try:
                os.remove(temp_env_path)
            except Exception as cleanup_err:
                print(f"[Warning] Failed to remove temporary file {temp_env_path}: {cleanup_err}")
        if "temp_env" in sys.modules:
            del sys.modules["temp_env"]

    # 3. Generate local artifacts
    print("[Agenlus] Exporting model to model.pt and model.onnx...")
    try:
        # Save PyTorch Model
        torch.save(model, "model.pt")
        
        # Determine dummy input shape (prefer local_obs_space detected during local evaluation)
        target_obs_space = local_obs_space if local_obs_space is not None else obs_space
        if isinstance(target_obs_space, (list, tuple)):
            dummy_input = torch.randn(1, *target_obs_space)
        elif isinstance(target_obs_space, int):
            dummy_input = torch.randn(1, target_obs_space)
        else:
            try:
                obs_int = int(target_obs_space)
                dummy_input = torch.randn(1, obs_int)
            except:
                dummy_input = torch.randn(1, 4) # fallback
                
        # Export ONNX
        model.eval()
        torch.onnx.export(
            model,
            dummy_input,
            "model.onnx",
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
        )
        print("[Agenlus] Local model conversion completed successfully.")
    except Exception as e:
        cleanup_local_files()
        err_msg = str(e)
        if "onnxscript" in err_msg or "No module named 'onnxscript'" in err_msg:
            raise ImportError(
                "Model export failed due to missing 'onnxscript' package. "
                "Please install it using: pip install onnxscript"
            ) from e
        elif "onnx" in err_msg or "No module named 'onnx'" in err_msg:
            raise ImportError(
                "Model export failed due to missing 'onnx' package. "
                "Please install it using: pip install onnx"
            ) from e
        raise Exception(f"Model export failed: {e}")

    # 4. Connect and Setup HuggingFace target
    try:
        from huggingface_hub import HfApi, create_repo
    except ImportError:
        cleanup_local_files()
        raise ImportError("huggingface_hub library is missing. Please install it with 'pip install huggingface_hub'.")
        
    print("[Agenlus] Connecting to HuggingFace...")
    try:
        api = HfApi(token=hf_token)
        user_info = api.whoami()
        hf_username = user_info.get("name")
        print(f"[Agenlus] Authenticated as HuggingFace user: {hf_username}")
    except Exception as e:
        cleanup_local_files()
        raise Exception(f"HuggingFace authentication failed: {e}")

    # Resolve HF Repo (User specified > User Profile Default setting > Fallback)
    if not hf_repo:
        if default_hf_repo:
            hf_repo = default_hf_repo
            print(f"[Agenlus] Using default repository from profile settings: {hf_repo}")
        else:
            hf_repo = f"{hf_username}/agenlus-agents"
            print(f"[Agenlus] No profile default repo found. Using default: {hf_repo}")
            
    # Normalize repo format (username/repo)
    if "/" not in hf_repo:
        hf_repo = f"{hf_username}/{hf_repo}"

    # Assign/Generate model name and subfolder
    if not model_name:
        clean_env_name = normalized_env_id.replace("system/", "").replace("/", "_").lower()
        model_name = f"model_{clean_env_name}_{int(time.time())}"
        
    subfolder = model_name.strip().replace(" ", "_").replace("/", "_")
    print(f"[Agenlus] Target repository: {hf_repo}, Subfolder: {subfolder}")
    
    try:
        create_repo(repo_id=hf_repo, token=hf_token, exist_ok=True, private=True)
    except Exception as e:
        print(f"[Agenlus] Skipped repository creation (proceeding anyway): {e}")

    # Create local temporary readme for this specific model subfolder
    readme_content = f"""---
tags:
- reinforcement-learning
- agenlus
- env-id: {normalized_env_id}
---

# RL Model: {model_name}
Trained locally and uploaded to Agenlus.

- **Environment**: `{normalized_env_id}`
- **Best Score**: {best_score}
- **Training Episodes**: {episodes}
"""
    
    with open("README.md", "w", encoding="utf-8") as f:
        f.write(readme_content)
        
    # Upload files under the specified subfolder path (stacking)
    print(f"[Agenlus] Uploading files to Hugging Face Hub under subfolder: {subfolder}...")
    try:
        api.upload_file(
            path_or_fileobj="model.pt",
            path_in_repo=f"{subfolder}/model.pt",
            repo_id=hf_repo
        )
        api.upload_file(
            path_or_fileobj="model.onnx",
            path_in_repo=f"{subfolder}/model.onnx",
            repo_id=hf_repo
        )
        api.upload_file(
            path_or_fileobj="README.md",
            path_in_repo=f"{subfolder}/README.md",
            repo_id=hf_repo
        )
        print("[Agenlus] HuggingFace files uploaded successfully.")
    except Exception as e:
        cleanup_local_files()
        raise Exception(f"Failed to upload files to Hugging Face: {e}")

    # 5. Register model on Agenlus Leaderboard
    print("[Agenlus] Registering model on Agenlus leaderboard...")
    register_url = f"{api_url}/api/rl/apps/register"
    reg_headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json"
    }
    
    hf_url = f"https://huggingface.co/{hf_repo}/tree/main/{subfolder}"
    
    payload = {
        "name": model_name,
        "description": f"Locally trained PyTorch model stacked for {normalized_env_id}",
        "environmentId": normalized_env_id,
        "bestScore": best_score,
        "algorithm": "PyTorch",
        "episodes": episodes,
        "huggingFaceRepo": hf_repo,
        "huggingFaceUrl": hf_url,
        "metadata": {
            "isLocalUpload": True,
            "seed": seed,
            "seedPinned": True if seed is not None else False
        }
    }
    
    try:
        reg_response = requests.post(register_url, headers=reg_headers, json=payload)
        if reg_response.status_code not in (200, 201):
            raise Exception(f"Leaderboard registration failed: {reg_response.text} (Status: {reg_response.status_code})")
        
        reg_data = reg_response.json()
        print(f"[Success] Model successfully registered on Agenlus leaderboard!")
        print(f"[Success] Model ID: {reg_data.get('id')}")
        print(f"[Success] Leaderboard HF URL: {hf_url}")
    except Exception as e:
        cleanup_local_files()
        raise Exception(f"Failed to register model on Agenlus server: {e}")
        
    cleanup_local_files()

def download(env_id: str, save_path: str = None):
    """
    Download the custom environment Python source code from Agenlus and save it locally.
    """
    if not os.path.exists(CONFIG_FILE):
        raise ValueError("No login session found. Please call agenlus.login(token) first.")
        
    try:
        with open(CONFIG_FILE, "r", encoding="utf-8") as f:
            config = json.load(f)
    except Exception as e:
        raise ValueError(f"Failed to read config file: {e}")
        
    token = config.get("token")
    api_url = config.get("api_url", DEFAULT_API_URL)
    
    headers = {}
    if token:
        headers["Authorization"] = f"Bearer {token}"
        
    print(f"[Agenlus] Resolving environment specs for {env_id}...")
    normalized_env_id = env_id
    name = env_id.split('/')[-1]
    
    # 1. Resolve normalized env ID via server spec endpoint
    try:
        spec_url = f"{api_url}/api/rl/environments/spec/{env_id}"
        response = requests.get(spec_url, headers=headers)
        if response.status_code == 200:
            env_spec_data = response.json()
            normalized_env_id = env_spec_data.get("normalizedEnvId", env_id)
            name = env_spec_data.get("name", name)
    except Exception as e:
        print(f"[Warning] Failed to resolve normalized ID: {e}. Attempting direct download with raw ID.")

    print(f"[Agenlus] Downloading source code for {normalized_env_id}...")
    
    # 2. Call appropriate endpoint based on layout
    if "/" in normalized_env_id:
        username, env_name = normalized_env_id.split("/")
        url = f"{api_url}/api/rl/environments/u/{username}/{env_name}"
    else:
        url = f"{api_url}/api/rl/environments/store/{normalized_env_id}"
        
    try:
        response = requests.get(url, headers=headers)
        if response.status_code != 200:
            raise Exception(f"Server returned status {response.status_code}: {response.text}")
            
        data = response.json()
        code = data.get("code")
        
        if not code:
            raise ValueError(f"Environment source code is empty or missing from server response.")
            
        if not save_path:
            save_path = f"{name}.py"
            
        with open(save_path, "w", encoding="utf-8") as f:
            f.write(code)
            
        print(f"[Success] Environment '{normalized_env_id}' source code saved to {save_path} ✅")
        
        if normalized_env_id.startswith("system/"):
            print("[Info] 💡 Built-in environment detected.")
            print("       Local rendering (env.render()) is disabled or may fail with an ImportError")
            print("       because it is optimized with bindings specific to the web simulator runtime.")
        else:
            print("[Info] 💡 Custom environment detected.")
            print("       This environment has self-contained rendering logic and can be rendered")
            print("       locally once its required graphical dependencies (e.g., pygame) are installed.")
            
        return save_path
    except Exception as e:
        raise Exception(f"Failed to download environment code: {e}")

def cleanup_local_files():
    for f in ["model.pt", "model.onnx", "README.md"]:
        if os.path.exists(f):
            try:
                os.remove(f)
            except Exception as e:
                print(f"[Warning] Failed to cleanup {f}: {e}")
