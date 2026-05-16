import time
import itertools
from google import genai
from google.genai import types
# from google.genai.services import ModelService

class RotatingGeminiClient:
    def __init__(self, api_keys: list, model_name: str = "gemini-2.0-flash"):
        """
        Use gemini-2.0-flash-exp (no thinking tokens) for reliable short answers.
        """
        # print(ModelService.ListModels())
        self.keys = [k for k in api_keys if k]
        if not self.keys:
            raise ValueError("No Gemini API keys provided")
        self.key_cooldown_until = {key: 0 for key in self.keys}
        self.key_cycle = itertools.cycle(self.keys)
        self.model_name = model_name
        self.current_key = next(self.key_cycle)
        self.client = genai.Client(api_key=self.current_key)
        print(f"[Gemini] Initialized with {len(self.keys)} keys, model={self.model_name}")

    def _get_next_available_key(self):
        start_key = self.current_key
        while True:
            if time.time() > self.key_cooldown_until.get(self.current_key, 0):
                return self.current_key
            self.current_key = next(self.key_cycle)
            if self.current_key == start_key:
                min_cooldown = min(self.key_cooldown_until.values())
                wait_time = max(0, min_cooldown - time.time()) + 1
                print(f"[Gemini] All keys cooling. Waiting for {wait_time:.2f} seconds...")
                time.sleep(wait_time)
                start_key = self.current_key

    def generate(self, prompt: str, max_attempts_per_key: int = 3) -> str:
        for attempt in range(max_attempts_per_key):
            self.current_key = self._get_next_available_key()
            try:
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        temperature=0.0,
                        max_output_tokens=100,   # enough for a short phrase
                    )
                )
                # Extract text from the first candidate
                if response.candidates and len(response.candidates) > 0:
                    candidate = response.candidates[0]
                    if candidate.content and candidate.content.parts:
                        text_parts = [p.text for p in candidate.content.parts if p.text]
                        if text_parts:
                            full_text = "".join(text_parts).strip()
                            return full_text
                # If no text, log and rotate key
                print(f"[Gemini] Empty response for key {self.current_key[-5:]}. Response: {response}")
                self.key_cooldown_until[self.current_key] = time.time() + 30
                continue
            except Exception as e:
                error_msg = str(e).lower()
                if "429" in error_msg or "quota" in error_msg or "rate limit" in error_msg:
                    cooldown_seconds = 30 * (2 ** attempt)
                    print(f"[Gemini] Key {self.current_key[-5:]} rate limited. Cooling for {cooldown_seconds}s.")
                    self.key_cooldown_until[self.current_key] = time.time() + cooldown_seconds
                    continue
                # For other errors (404, etc.), rotate key
                print(f"[Gemini] Error with key {self.current_key[-5:]}: {e}. Rotating...")
                self.key_cooldown_until[self.current_key] = time.time() + 30
                continue
        raise Exception("All Gemini keys exhausted or failed after multiple attempts")