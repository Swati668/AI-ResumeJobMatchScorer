import hashlib
import json

def create_cache_key(agent_name: str, context: dict) -> str:
    """
    Creates a unique key based on:
    - agent name
    - input context
    """
    normalized = json.dumps(context, sort_keys=True)
    raw_key = f"{agent_name}:{normalized}"

    return hashlib.md5(raw_key.encode()).hexdigest()