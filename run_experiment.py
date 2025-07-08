"""
This module is used to access Langsmith and to process natural language cases with the configured LLM.
"""

import asyncio
from langsmith import Client
from langchain_core.rate_limiters import InMemoryRateLimiter
import os
from dotenv import load_dotenv


load_dotenv()
os.environ["LANGSMITH_API_KEY"] = os.getenv("LANGSMITH_API_KEY")
os.environ["LANGSMITH_TRACING"] = os.getenv("LANGSMITH_TRACING")
os.environ["LANGSMITH_ENDPOINT"] = os.getenv("LANGSMITH_ENDPOINT")

# Rate Limiter konfigurieren (z.B. 1 Request alle 10 Sekunden)
rate_limiter = InMemoryRateLimiter(
    requests_per_second=10,  # 0.1 <-- Super slow! We can only make a request once every 10 seconds!!
    check_every_n_seconds=3, # 0.1 Wake up every 100 ms to check whether allowed to make a request,
    max_bucket_size=100000
)

client = Client(api_key=os.getenv("LANGSMITH_API_KEY"))
datasets = {"M&W","Dupont","Real-world","Exam"}

async def main():
    #for set in datasets:
    chain = client.pull_prompt("voo5", include_model=True)

    dataset = client.read_dataset(dataset_name="Dupont")

    rate_limiter.acquire()

    await client.aevaluate(chain, data=dataset)
        #print(result)

# Programm ausführen
asyncio.run(main())