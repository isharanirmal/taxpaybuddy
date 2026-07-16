import os
import time
from dotenv import load_dotenv

load_dotenv()


from google import genai
from google.genai import types
from google.genai.errors import APIError

from src.framework.interfaces.interfaces import ILLMClient


class GeminiClient(ILLMClient):
    """
    Gemini API client with modern SDK and auto-retry logic.
    """

    def __init__(self):
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("Gemini API key not found.")

        self.client = genai.Client(api_key=api_key)
        self.model_name = "gemini-flash-latest"

    def generate(self, prompt: str, system_instruction: str = "") -> str:
        config = types.GenerateContentConfig(
            system_instruction=system_instruction if system_instruction else None
        )

        retries = 3
        delay = 6  

        for i in range(retries):
            try:
               
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=prompt,
                    config=config
                )
                return response.text
                
            except APIError as e:
                
                if e.code in (429, 503):
                    if i == retries - 1:  
                        raise e
                    
                    print(f"\n[⚠️ RATE LIMIT] out {delay}try again...")
                    time.sleep(delay)
                    delay *= 2 
                else:
                    
                    raise e