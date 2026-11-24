from llm_safety_evaluator import *
import json
print(json.dumps(evaluate(synthetic_cases(),lambda p:"hello" if "hello" in p else "Cannot disclose"),indent=2))
