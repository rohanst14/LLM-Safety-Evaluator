"""Callable-adapter regression example, separate from measured benchmarks."""
import json
from llm_safety_evaluator import evaluate,synthetic_cases

if __name__=="__main__":
    print(json.dumps(evaluate(synthetic_cases(),lambda p:"hello" if "hello" in p else "Cannot disclose"),indent=2))
