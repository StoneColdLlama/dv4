"""
dv4/data/prepare_data.py

Downloads and prepares training data for the DV4 4-topic run.

Topic 0 — Math:    GSM8K
Topic 1 — General: TriviaQA rc.nocontext
Topic 2 — Code:    CodeSearchNet (Python docstrings + code)
Topic 3 — Science: SciQ (science exam questions)

Output: dv4/data/{math,general,code,science}.jsonl

Usage:
    python dv4/data/prepare_data.py
    python dv4/data/prepare_data.py --debug        # 500 samples each
    python dv4/data/prepare_data.py --samples 5000 # custom count

Author: Peter Norman / twoswans.com.au
"""

import sys
import json
import argparse
from pathlib import Path


def check_dependencies():
    missing = []
    try:
        import datasets
    except ImportError:
        missing.append("datasets")
    try:
        import huggingface_hub
    except ImportError:
        missing.append("huggingface_hub")
    if missing:
        print("Missing packages — install with:")
        print(f"  pip install {' '.join(missing)}")
        sys.exit(1)
    import datasets as ds
    print(f"Dependencies OK (datasets {ds.__version__})")


def save_jsonl(samples, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        for text in samples:
            f.write(json.dumps({"text": text}) + "\n")
    print(f"  Saved {len(samples)} samples → {path}")


def prepare_math(output_path, max_samples):
    from datasets import load_dataset
    print("\n[Topic 0 — Math] Downloading GSM8K...")
    ds = load_dataset("openai/gsm8k", "main", split="train")
    samples = []
    for item in ds:
        if len(samples) >= max_samples:
            break
        samples.append(f"Question: {item['question']}\nAnswer: {item['answer']}")
    save_jsonl(samples, output_path)


def prepare_general(output_path, max_samples):
    from datasets import load_dataset
    print("\n[Topic 1 — General] Downloading TriviaQA...")
    ds = load_dataset("trivia_qa", "rc.nocontext",
                      split=f"train[:{max_samples}]", trust_remote_code=True)
    samples = []
    for item in ds:
        answer = item["answer"]["value"]
        if answer.strip():
            samples.append(f"Question: {item['question']}\nAnswer: {answer}")
    save_jsonl(samples, output_path)


def prepare_code(output_path, max_samples):
    from datasets import load_dataset
    print("\n[Topic 2 — Code] Downloading CodeSearchNet (Python)...")
    ds = load_dataset("code_search_net", "python",
                      split=f"train[:{max_samples}]", trust_remote_code=True)
    samples = []
    for item in ds:
        if len(samples) >= max_samples:
            break
        docstring = item.get("func_documentation_string", "").strip()
        code = item.get("func_code_string", "").strip()
        if docstring and code:
            samples.append(f"# {docstring}\n{code}")
        elif code:
            samples.append(code)
    save_jsonl(samples, output_path)


def prepare_science(output_path, max_samples):
    from datasets import load_dataset
    print("\n[Topic 3 — Science] Downloading SciQ...")
    ds = load_dataset("sciq", split=f"train[:{max_samples}]")
    samples = []
    for item in ds:
        if len(samples) >= max_samples:
            break
        question = item.get("question", "").strip()
        answer = item.get("correct_answer", "").strip()
        support = item.get("support", "").strip()
        if question and answer:
            text = f"Question: {question}\nAnswer: {answer}"
            if support:
                text += f"\nExplanation: {support}"
            samples.append(text)
    save_jsonl(samples, output_path)


def validate_file(path, topic):
    if not path.exists():
        print(f"  ERROR: {path} not found")
        return False
    size_mb = path.stat().st_size / 1e6
    line_count = sum(1 for _ in open(path, encoding='utf-8'))
    with open(path, encoding='utf-8') as f:
        first = json.loads(f.readline())
    print(f"\n[Validate] {topic}")
    print(f"  Lines: {line_count}  Size: {size_mb:.1f}MB")
    print(f"  Sample: {first['text'][:80]}...")
    ok = "text" in first and len(first["text"]) > 0
    print(f"  Status: {'✅ OK' if ok else '❌ FAILED'}")
    return ok


def main():
    parser = argparse.ArgumentParser(description='Prepare DV4 4-topic training data')
    parser.add_argument('--samples',    type=int, default=7000,
                        help='Samples per topic (default: 7000)')
    parser.add_argument('--debug',      action='store_true',
                        help='500 samples per topic for fast testing')
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--topics',     type=str, default='0,1,2,3',
                        help='Which topics to download (default: 0,1,2,3)')
    args = parser.parse_args()

    print("DV4 4-Topic Data Preparation")
    print("=" * 60)
    check_dependencies()

    output_dir = Path(args.output_dir) if args.output_dir else Path(__file__).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    n = 500 if args.debug else args.samples
    topics = [int(t) for t in args.topics.split(',')]

    print(f"\nSamples per topic: {n}")
    print(f"Topics to prepare: {topics}")
    print(f"Output dir: {output_dir}")

    paths = {
        0: output_dir / "math.jsonl",
        1: output_dir / "general.jsonl",
        2: output_dir / "code.jsonl",
        3: output_dir / "science.jsonl",
    }

    preparers = {
        0: prepare_math,
        1: prepare_general,
        2: prepare_code,
        3: prepare_science,
    }

    for topic_id in topics:
        preparers[topic_id](paths[topic_id], n)

    print("\n" + "=" * 60)
    all_ok = all(validate_file(paths[t], f"Topic {t}") for t in topics)

    print("\n" + "=" * 60)
    if all_ok:
        print("Data preparation complete ✅")
        print("\nNext step — train the 500M model:")
        print(f"  python -m dv4.training.train \\")
        for t in topics:
            topic_names = {0:'math', 1:'general', 2:'code', 3:'science'}
            print(f"    --{topic_names[t]}_data {paths[t]} \\")
        print(f"    --output_dir /mnt/Models/dv4_500m_run1 \\")
        print(f"    --model_size medium \\")
        print(f"    --steps_per_topic {'100' if args.debug else '2000'}")
    else:
        print("Data preparation FAILED ❌")
        sys.exit(1)


if __name__ == "__main__":
    main()
