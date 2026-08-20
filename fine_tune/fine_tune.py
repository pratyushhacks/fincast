"""
fine_tune.py — QLoRA fine-tune of Phi-3-mini-4k-instruct on the FinCast
move_bin classification dataset, sized for a 6GB laptop GPU (RTX 4050).

Before running the full thing, do the two cheap sanity checks at the top
(token-length profiling + target_modules match + masking preview) — each
takes seconds and catches failure modes that otherwise waste a full,
possibly multi-hour training run before you notice anything's wrong.
"""
import os
import json
import torch
from dotenv import load_dotenv
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig, DataCollatorForCompletionOnlyLM
from datasets import load_dataset
from transformers import EarlyStoppingCallback

load_dotenv()  # reads .env in the current working directory

MODEL_ID = os.getenv("MODEL_ID", "microsoft/Phi-3-mini-4k-instruct")
TRAIN_PATH = os.getenv("TRAIN_PATH", "data/combined/train.jsonl")
VAL_PATH = os.getenv("VAL_PATH", "data/combined/val.jsonl")
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "./phi3-fincast-qlora")

print(f"Config: MODEL_ID={MODEL_ID}  TRAIN_PATH={TRAIN_PATH}  VAL_PATH={VAL_PATH}  OUTPUT_DIR={OUTPUT_DIR}")

# Phi-3's chat template renders assistant turns after this exact marker —
# used below to mask loss so the model is only trained to predict the
# move_bin JSON, not to reproduce the system/user prompt tokens.
RESPONSE_TEMPLATE = "<|assistant|>"


# ---------------------------------------------------------------------------
# Step 0 — quick dataset sanity check, no GPU required. Run this alone first.
# ---------------------------------------------------------------------------
def profile_dataset(tokenizer, path: str, label: str) -> None:
    lengths = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            record = json.loads(line)
            formatted = tokenizer.apply_chat_template(record["messages"], tokenize=False)
            n_tokens = len(tokenizer(formatted)["input_ids"])
            lengths.append(n_tokens)
            # Sanity check: RESPONSE_TEMPLATE must appear so the completion-
            # only collator can find where the assistant turn starts. If this
            # ever fails, the masking below would silently mask nothing (or
            # everything) instead of just the completion.
            assert RESPONSE_TEMPLATE in formatted, (
                f"'{RESPONSE_TEMPLATE}' not found in formatted example from {path} — "
                "check the tokenizer's chat_template matches what's assumed here."
            )

    lengths.sort()
    n = len(lengths)
    print(f"[{label}] n={n}  min={lengths[0]}  "
          f"p50={lengths[n//2]}  p95={lengths[int(n*0.95)]}  max={lengths[-1]}")


# ---------------------------------------------------------------------------
# Step 1 — model + tokenizer
# ---------------------------------------------------------------------------
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,  # extra ~0.4GB savings, worth it at 6GB
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
    attn_implementation="eager",  # flash-attn2 often won't build cleanly on 4050 laptop drivers; eager is safe
)
model.config.use_cache = False  # required alongside gradient checkpointing

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"  # required for causal LM training (not generation)

# --- Sanity check A: dataset token lengths, before committing to a max_seq_length ---
print("\n--- Token length profile (run this once before a full training run) ---")
profile_dataset(tokenizer, TRAIN_PATH, "train")
profile_dataset(tokenizer, VAL_PATH, "val")
print("Set max_seq_length below comfortably above the p95/max you see here.\n")

# --- Sanity check B: confirm LoRA target_modules actually exist on this model ---
# This is the exact silent-failure class we've hit before: wrong module
# names -> LoRA matches zero parameters -> training "succeeds" and trains
# nothing. Phi-3 fuses attention/MLP projections differently from most
# other model families (qkv_proj/gate_up_proj instead of separate
# q_proj/k_proj/v_proj/gate_proj/up_proj) — if you ever swap in a different
# base model (e.g. Qwen2.5), this list must change too.
target_modules = ["qkv_proj", "o_proj", "gate_up_proj", "down_proj"]
matched = [name for name, _ in model.named_modules() if any(name.endswith(t) for t in target_modules)]
print(f"--- LoRA target_modules sanity check: {len(matched)} matching modules found ---")
if not matched:
    raise RuntimeError(
        "No modules matched target_modules — LoRA would train nothing. "
        "Run `for name,_ in model.named_modules(): print(name)` to find the correct names."
    )

peft_config = LoraConfig(
    r=16,                    # 8 if you OOM, 32 if you have headroom to spare
    lora_alpha=32,           # 2x rank is a reasonable default
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=target_modules,
)

# ---------------------------------------------------------------------------
# Step 2 — data
# ---------------------------------------------------------------------------
train_dataset = load_dataset("json", data_files=TRAIN_PATH, split="train")
val_dataset = load_dataset("json", data_files=VAL_PATH, split="train")
print(f"Train: {len(train_dataset)} rows  |  Val: {len(val_dataset)} rows")

# Mask loss to the assistant's completion only — otherwise the model spends
# most of its gradient signal learning to reproduce the system/user prompt
# tokens instead of the few tokens of JSON that actually matter for this task.
collator = DataCollatorForCompletionOnlyLM(
    response_template=RESPONSE_TEMPLATE,
    tokenizer=tokenizer,
)

# ---------------------------------------------------------------------------
# Step 3 — training config
# ---------------------------------------------------------------------------
# With 1,068 train rows, batch_size=1, grad_accum=16 -> effective batch 16 ->
# ~67 optimizer steps/epoch. 3 epochs (~200 steps) is likely too few for a
# LoRA adapter with zero prior exposure to this exact task. Using more
# epochs + early stopping on eval_loss instead of guessing a fixed count —
# safer for a dataset this size, since it stops automatically if it starts
# overfitting rather than requiring you to pre-guess the right number.
sft_config = SFTConfig(
    output_dir=OUTPUT_DIR,
    max_seq_length=768,          # right-size this after checking profile_dataset() output above
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=16,     # effective batch size 16
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    num_train_epochs=8,                 # upper bound — early stopping will likely stop sooner
    learning_rate=2e-4,
    lr_scheduler_type="cosine",
    warmup_ratio=0.03,
    optim="paged_adamw_8bit",           # crucial at 6GB — regular adamw_8bit still spikes on paging
    logging_steps=10,
    eval_strategy="steps",
    eval_steps=25,
    save_strategy="steps",
    save_steps=25,
    save_total_limit=3,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    bf16=True,
    report_to="none",
    packing=False,                       # must stay off — packing would break completion-only masking
                                          # (response_template search assumes one example per sequence)
)

trainer = SFTTrainer(
    model=model,
    args=sft_config,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    peft_config=peft_config,
    tokenizer=tokenizer,
    data_collator=collator,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=4)],  # ~4 evals (100 steps) with no improvement -> stop
)

# --- Sanity check C: print one formatted + masked example before training ---
# Confirms the collator is actually masking the prompt and keeping the
# assistant JSON as the only trainable target. Cheap, worth eyeballing once.
sample_batch = collator([trainer.train_dataset[0]])
print("\n--- Masking check on example 0 (labels == -100 means masked out of loss) ---")
print("input_ids decoded:", tokenizer.decode(sample_batch["input_ids"][0]))
label_ids = sample_batch["labels"][0]
visible = [tok for tok in label_ids.tolist() if tok != -100]
print("Tokens NOT masked (i.e. actually trained on):", tokenizer.decode(visible))

if __name__ == "__main__":
    trainer.train()
    trainer.save_model(f"{OUTPUT_DIR}-final")
    tokenizer.save_pretrained(f"{OUTPUT_DIR}-final")
    print(f"\nDone. Adapter saved to {OUTPUT_DIR}-final")