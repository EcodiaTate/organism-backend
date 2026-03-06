import asyncio
import json
import os

import redis.asyncio as redis
from dotenv import load_dotenv

# Force Python to read your .env file
load_dotenv()

async def fire_signal():
    url = os.getenv("ECODIAOS_REDIS__URL", "redis://localhost:6379/0")
    password = os.getenv("ECODIAOS_REDIS_PASSWORD")

    # Let's print this just to prove it actually found the password this time
    print(f"[*] Loaded password: {'[HIDDEN]' if password else 'MISSING!'}")

    r = redis.from_url(url, password=password)
    payload = json.dumps({"files_changed": ["systems/atune/logging_utils.py"]})

    print(f"[*] Firing neurotransmitter to {url}...")
    subscribers = await r.publish("eos:events:code_evolved", payload)

    print(f"[+] Signal fired! Heard by {subscribers} receptor(s).")
    await r.aclose()

if __name__ == "__main__":
    asyncio.run(fire_signal())
