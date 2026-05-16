import os
import json
import torch
from safetensors.torch import load_file
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.configs.policies import PreTrainedConfig

# ---------- 1. 检查 CUDA ----------
if not torch.cuda.is_available():
    raise RuntimeError("❌ CUDA 不可用！请在终端执行：\n"
                       "pip uninstall torch torchvision torchaudio -y\n"
                       "pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121")
device = torch.device("cuda")
print(f"使用设备: {device}")

# ---------- 2. 本地模型文件夹 ----------
model_dir = "./smolvla_model"
config_path = os.path.join(model_dir, "train_config.json")
weights_path = os.path.join(model_dir, "model.safetensors")
preprocessor_path = os.path.join(model_dir, "preprocessor_config.json")

# 安全检查
for p in [config_path, weights_path, preprocessor_path]:
    if not os.path.exists(p):
        raise FileNotFoundError(f"找不到文件: {p}")

# ---------- 3. 手动加载配置 ----------
with open(config_path, "r") as f:
    config_dict = json.load(f)

# 必须包含 policy 部分，否则补全
if "policy" not in config_dict:
    config_dict["policy"] = {
        "type": "smolvla",
        "pretrained_path": model_dir,
        "device": "cuda",
        "use_amp": False,
        "train_expert_only": True,
    }

# 构造 PreTrainedConfig
config = PreTrainedConfig.from_dict(config_dict)

# ---------- 4. 直接实例化策略 ----------
policy = SmolVLAPolicy(config)
policy.to(device)
policy.eval()

# 加载微调权重
state_dict = load_file(weights_path)
policy.load_state_dict(state_dict, strict=False)
print("✅ 模型加载成功！")

# ---------- 5. 模拟输入 ----------
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained(model_dir)

dummy_obs = {
    "observation.images.top": torch.randn(1, 3, 512, 512, device=device),
    "observation.images.wrist": torch.randn(1, 3, 512, 512, device=device),
    "observation.state": torch.randn(1, 6, device=device),
    "observation.task": tokenizer(
        "Pick up the red cube and place it in the blue box",
        return_tensors="pt"
    )["input_ids"].to(device)
}

with torch.no_grad():
    action = policy(dummy_obs)

print(f"✅ 推理成功！生成的动作形状: {action.shape}")
print(f"示例动作（前5步，第1个关节）: {action[0, :5, 0].cpu().numpy()}")