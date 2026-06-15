import json
import re
from collections import Counter
from transformers import AutoTokenizer

DATA_PATH = "data/train.jsonl"
MODEL_NAME = "microsoft/phi-2"

def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    with open(DATA_PATH, "r") as f:
        lines = f.readlines()
        
    examples = [json.loads(line) for line in lines]
    
    print(f"Total Examples: {len(examples)}")
    
    # 1. Duplicates
    texts = [ex["text"] for ex in examples]
    unique_texts = set(texts)
    print(f"Unique Examples: {len(unique_texts)}")
    print(f"Exact Duplicates: {len(texts) - len(unique_texts)}")
    
    # 2. Token Lengths
    token_lengths = [len(tokenizer.encode(t)) for t in texts]
    avg_len = sum(token_lengths) / len(token_lengths)
    max_len = max(token_lengths)
    min_len = min(token_lengths)
    print(f"Average Token Length: {avg_len:.1f}")
    print(f"Min Token Length: {min_len}")
    print(f"Max Token Length: {max_len}")
    
    # 3. Analyze Schema & Structure
    # The dataset uses a simple string block. We can regex extract parts.
    # Typical:
    # <s>[INST] ... [/INST]
    # Symptom: ...
    # Possible Causes: ...
    # Risk Level: ...
    # Safe Checks: ...
    # Recommendation: ...</s>
    
    missing_symptom = 0
    missing_causes = 0
    missing_risk = 0
    missing_checks = 0
    missing_rec = 0
    
    risk_levels = Counter()
    symptoms = Counter()
    causes_counter = Counter()
    
    inputs_to_outputs = {}
    contradictions = 0
    
    for ex in examples:
        text = ex["text"]
        
        # Split into input and output
        if "[/INST]" in text:
            inp, out = text.split("[/INST]", 1)
        else:
            inp, out = text, ""
            
        inp = inp.strip()
        out = out.strip().replace("</s>", "")
        
        # Check contradictions (same input, different output)
        if inp in inputs_to_outputs:
            if inputs_to_outputs[inp] != out:
                contradictions += 1
        else:
            inputs_to_outputs[inp] = out
            
        # Check Schema
        has_symp = "Symptom:" in out
        has_cause = "Possible Causes:" in out
        has_risk = "Risk Level:" in out
        has_checks = "Safe Checks:" in out
        has_rec = "Recommendation:" in out
        
        if not has_symp: missing_symptom += 1
        if not has_cause: missing_causes += 1
        if not has_risk: missing_risk += 1
        if not has_checks: missing_checks += 1
        if not has_rec: missing_rec += 1
        
        # Extract Risk Level
        risk_match = re.search(r"Risk Level:\s*(.*?)\n", out + "\n")
        if risk_match:
            risk = risk_match.group(1).strip().strip(".")
            risk_levels[risk] += 1
            
        # Extract Symptom
        symp_match = re.search(r"Symptom:\s*(.*?)\n", out + "\n")
        if symp_match:
            symptoms[symp_match.group(1).strip()] += 1
            
        # Extract Causes (first one only for simplicity)
        cause_match = re.search(r"Possible Causes:\s*(.*?)\n", out + "\n")
        if cause_match:
            cause_line = cause_match.group(1).strip()
            # Just take the first few words or the whole line
            causes_counter[cause_line[:30]] += 1
            
    print("--- Schema Adherence ---")
    print(f"Missing 'Symptom:': {missing_symptom}")
    print(f"Missing 'Possible Causes:': {missing_causes}")
    print(f"Missing 'Risk Level:': {missing_risk}")
    print(f"Missing 'Safe Checks:': {missing_checks}")
    print(f"Missing 'Recommendation:': {missing_rec}")
    
    print(f"Contradictions (Same input, diff output): {contradictions}")
    
    print("--- Risk Levels ---")
    for r, c in risk_levels.most_common(10):
        print(f"  {r}: {c}")
        
    print("--- Top 5 Symptoms ---")
    for s, c in symptoms.most_common(5):
        print(f"  {s}: {c}")

if __name__ == "__main__":
    main()
