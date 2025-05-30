## Where ollama mistral will be located/initiated 
import subprocess
import os
from dotenv import load_dotenv

load_dotenv()

OLLAMA = os.getenv(r"OLLAMA_PATH")

def query_local_llm(prompt: str, model: str = "mistral") -> str:
    try:
        result = subprocess.run(
            [OLLAMA, "run", model],
            input=prompt,
            text=True,
            ## Handles encoding issues. without this you will see errors in concole
            encoding='utf-8',
            errors="replace",
            capture_output=True,
            timeout=60
        )
        return result.stdout.strip()
    except Exception as e:
        return f"Error querying local LLM: {e}"