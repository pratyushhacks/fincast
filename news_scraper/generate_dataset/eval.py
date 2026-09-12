"""
eval.py — generation-based evaluation of the fine-tuned FinCast adapter.

Loads the base model + LoRA adapter, generates a move_bin prediction for
every row in val.jsonl, and compares against ground truth. Reports:
  - overall accuracy vs. the actual val-set majority-class baseline
    (NOT the pre-fix 43.5% figure — that's stale; use whatever your own
    combine_datasets.py printed for this val split, e.g. ~31.2%)
  - confusion matrix (including an "invalid_json" bucket — a model that
    outputs unparseable text is a different failure mode than one that
    outputs valid JSON with the wrong class)
  - per-ticker accuracy
"""
import os
import json
import re
from collections import Counter, defaultdict

import torch
from dotenv import load_dotenv
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

# Anchored to this file's own location, not the current working directory —
# a bare load_dotenv() depends on cwd at runtime, which is exactly what
# caused the ADAPTER_PATH resolution bug (nightly_run.py's subprocess can
# have a different cwd than expected depending on how/where it's launched).
load_dotenv(dotenv_path=os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))

MODEL_ID = os.getenv("MODEL_ID")
VAL_PATH = os.getenv("VAL_PATH")
ADAPTER_PATH = os.getenv("ADAPTER_PATH")

print(f"Using MODEL_ID={MODEL_ID}, VAL_PATH={VAL_PATH}, ADAPTER_PATH={ADAPTER_PATH}")

MOVE_BINS = ["strong_down", "down", "flat", "up", "strong_up"]


def get_ticker(record: dict) -> str:
    for line in record["messages"][1]["content"].split("\n"):
        if line.startswith("Ticker: "):
            return line[len("Ticker: "):]
    return "UNKNOWN"


def get_true_move_bin(record: dict) -> str:
    return json.loads(record["messages"][2]["content"]).get("move_bin", "unknown")


def extract_move_bin(generated_text: str) -> str | None:
    """Parse the model's generated text for a move_bin value. Tries strict
    JSON first, falls back to a regex scan — a model that's still learning
    the exact format might emit something almost-JSON (missing a quote,
    trailing comma, etc.), and we want to credit it as "got the right class
    but imperfect formatting" separately from truly nonsensical output.
    """
    try:
        parsed = json.loads(generated_text.strip())
        val = parsed.get("move_bin")
        if val in MOVE_BINS:
            return val
    except (json.JSONDecodeError, AttributeError):
        pass

    match = re.search(r'"move_bin"\s*:\s*"(\w+)"', generated_text)
    if match and match.group(1) in MOVE_BINS:
        return match.group(1)

    # Last resort: does a valid bin name appear anywhere in the output at all
    for b in MOVE_BINS:
        if b in generated_text:
            return b
    return None


def load_val_records(path: str) -> list[dict]:
    records = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def build_prompt(tokenizer, record: dict) -> str:
    # Only system + user turns — add_generation_prompt appends the
    # "<|assistant|>\n" marker so the model knows to continue from there,
    # without ever seeing the ground-truth assistant content.
    prompt_messages = record["messages"][:2]
    return tokenizer.apply_chat_template(
        prompt_messages, tokenize=False, add_generation_prompt=True
    )


def load_model_and_tokenizer(model_id: str = MODEL_ID, adapter_path: str = ADAPTER_PATH):
    """Load the 4-bit base model + LoRA adapter, shared by eval.py and
    predict_ticker.py so both use the identical inference setup — no
    separately-drifting copies of the quantization config."""
    resolved_adapter_path = os.path.abspath(adapter_path)
    
    print(f"Loading base model {model_id} + adapter from {adapter_path} "
          f"(resolved to: {resolved_adapter_path}) ...")

    if not os.path.exists(os.path.join(resolved_adapter_path, "adapter_config.json")):
        raise FileNotFoundError(
            f"No adapter_config.json found at {resolved_adapter_path}. "
            f"This is almost always a relative-path problem: '{adapter_path}' resolves "
            f"relative to the current working directory at runtime, not to where the "
            f"adapter was actually saved. Fix by setting an ABSOLUTE path in .env, e.g.:\n"
            f"  ADAPTER_PATH=C:\\full\\path\\to\\phi3-fincast-qlora-final"
        )

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    base_model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="eager",
    )
    model = PeftModel.from_pretrained(base_model, resolved_adapter_path)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def main():
    model, tokenizer = load_model_and_tokenizer()

    val_records = load_val_records(VAL_PATH)
    print(f"Evaluating on {len(val_records)} val examples\n")

    y_true, y_pred, tickers = [], [], []
    invalid_count = 0

    for i, record in enumerate(val_records):
        prompt = build_prompt(tokenizer, record)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=20,   # the target is a few tokens of JSON — no need for more
                do_sample=False,     # greedy decoding for deterministic, comparable eval
                pad_token_id=tokenizer.eos_token_id,
            )

        generated = tokenizer.decode(
            output_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
        )
        pred = extract_move_bin(generated)
        true = get_true_move_bin(record)

        if pred is None:
            invalid_count += 1
            pred = "invalid_json"

        y_true.append(true)
        y_pred.append(pred)
        tickers.append(get_ticker(record))

        if (i + 1) % 20 == 0:
            print(f"  ...{i+1}/{len(val_records)} done")

    # --- Overall accuracy vs. the real majority-class baseline for this val set ---
    baseline_class, baseline_count = Counter(y_true).most_common(1)[0]
    baseline_acc = baseline_count / len(y_true)
    correct = sum(1 for t, p in zip(y_true, y_pred) if t == p)
    accuracy = correct / len(y_true)

    print("\n" + "=" * 60)
    print("OVERALL RESULTS")
    print("=" * 60)
    print(f"Val set size:            {len(y_true)}")
    print(f"Invalid/unparseable output: {invalid_count} ({invalid_count/len(y_true)*100:.1f}%)")
    print(f"Majority-class baseline: {baseline_acc*100:.1f}% (always predict '{baseline_class}')")
    print(f"Model accuracy:          {accuracy*100:.1f}%")
    print(f"Lift over baseline:      {(accuracy-baseline_acc)*100:+.1f} pts")

    # --- Confusion matrix ---
    labels = MOVE_BINS + ["invalid_json"]
    print("\n--- Confusion matrix (rows=true, cols=predicted) ---")
    header = "true\\pred".ljust(14) + "".join(l[:10].rjust(12) for l in labels)
    print(header)
    for t in MOVE_BINS:
        row_counts = Counter(p for tt, p in zip(y_true, y_pred) if tt == t)
        row = t.ljust(14) + "".join(str(row_counts.get(l, 0)).rjust(12) for l in labels)
        print(row)

    # --- Per-ticker accuracy ---
    per_ticker_correct = defaultdict(int)
    per_ticker_total = defaultdict(int)
    for t, p, tick in zip(y_true, y_pred, tickers):
        per_ticker_total[tick] += 1
        if t == p:
            per_ticker_correct[tick] += 1

    print("\n--- Per-ticker accuracy ---")
    for tick in sorted(per_ticker_total, key=lambda k: -per_ticker_total[k]):
        n = per_ticker_total[tick]
        acc = per_ticker_correct[tick] / n
        print(f"  {tick:<8} {acc*100:>5.1f}%  (n={n})")

    # Save raw predictions for further analysis
    out_path = "eval_predictions.jsonl"
    with open(out_path, "w", encoding="utf-8") as fh:
        for t, p, tick in zip(y_true, y_pred, tickers):
            fh.write(json.dumps({"ticker": tick, "true": t, "pred": p}) + "\n")
    print(f"\nSaved raw predictions to {out_path}")


if __name__ == "__main__":
    main()