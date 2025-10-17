import gc
import os
import time
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from sae_lens import SAE, HookedSAETransformer

# These must be set before importing any Hugging Face or sae_lens modules
os.environ["TRANSFORMERS_CACHE"] = "/shared/3/projects/bangzhao/models/transformers"
os.environ["HF_DATASETS_CACHE"] = "/shared/3/projects/bangzhao/models/datasets"
os.environ["HF_TOKENIZERS_CACHE"] = "/shared/3/projects/bangzhao/models/tokenizers"
os.environ["HF_HOME"] = "/shared/3/projects/bangzhao/models"  # fallback root

from huggingface_hub import login
login(token="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")


# -------- configs --------
device = "cuda:0" if torch.cuda.is_available() else "cpu"
MODEL_ID = "google/gemma-2-2b"
SAE_RELEASE = "gemma-scope-2b-pt-res-canonical"
INPUT_CSV = "/home/bangzhao/projects/saes-emotion/emo-llm-main/outputs/gemma-2-2b/filtered_train_data.csv"
SAVE_ROOT = "/shared/3/projects/bangzhao/interpretability/gemma-2-2b/SAE-activations"

# -------- load model --------
model = HookedSAETransformer.from_pretrained(MODEL_ID, device=device)
num_layers = model.cfg.n_layers
print(f"Loaded {MODEL_ID} with {num_layers} layers.")

# -------- load data --------
df = pd.read_csv(INPUT_CSV)

EMOTIONS = [
    "anger","boredom","disgust","fear","guilt","joy",
    "pride","relief","sadness","shame","surprise","trust","neutral"
]
emo2id = {e: i for i, e in enumerate(EMOTIONS)}

def norm_lbl(s: str) -> str:
    s = (s or "").strip().lower()
    if s in {"no-emotion", "no emotion"}:
        return "neutral"
    return s

df_ = df.copy()
df_["gold_emotion"] = df_["emotion"].astype(str).map(norm_lbl)
unknown = sorted(set(df_["gold_emotion"]) - set(EMOTIONS))
if unknown:
    print("Unknown labels in df (dropping):", unknown)
    df_ = df_[~df_["gold_emotion"].isin(unknown)]

y = torch.tensor(df_["gold_emotion"].map(emo2id).values, dtype=torch.long)
texts = df_["input_text"].astype(str).tolist()

# -------- SAE feature extraction --------
@torch.inference_mode()
def encode_sae_codes(layer, text_list, batch_size=4):
    sae_id = f"layer_{layer}/width_16k/canonical"
    sae, cfg_dict, sparsity = SAE.from_pretrained(
        release=SAE_RELEASE,
        sae_id=sae_id,
        device=device,
    )

    HOOK = f"blocks.{layer}.hook_resid_post.hook_sae_acts_post"
    model.eval()
    feats = []

    for i in tqdm(range(0, len(text_list), batch_size),
                  desc=f"Encoding SAE features @ layer {layer}", unit="batch"):
        batch_prompts = text_list[i:i+batch_size]
        _, cache = model.run_with_cache_with_saes(
            batch_prompts,
            saes=[sae],
            names_filter=lambda n: n == HOOK,
        )
        a = cache[HOOK][:, -1, :].to("cpu", non_blocking=True).contiguous()
        feats.append(a)

        # cleanup GPU
        del cache, a, batch_prompts
        torch.cuda.empty_cache()
        gc.collect()

    return torch.cat(feats, dim=0)  # [N, m]

# -------- main loop over layers --------
for layer in range(num_layers):
    print(f"\n===== Processing Layer {layer} =====")

    try:
        X = encode_sae_codes(layer, texts, batch_size=4)
        print(f"Layer {layer} done: {X.shape}, labels = {y.shape}")

        save_dir = os.path.join(SAVE_ROOT, f"gemma-2-2B-layer-{layer}")
        os.makedirs(save_dir, exist_ok=True)

        np.save(os.path.join(save_dir, "X.npy"), X.cpu().numpy())
        np.save(os.path.join(save_dir, "y.npy"), y.cpu().numpy())

        # cleanup memory after each layer
        del X
        torch.cuda.empty_cache()
        gc.collect()

    except Exception as e:
        print(f"Error on layer {layer}: {e}")
        continue

print("\n All layers processed and saved.")
