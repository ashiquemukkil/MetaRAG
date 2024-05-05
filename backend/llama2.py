from langchain.chat_models import ChatOllama
from langchain.embeddings import OllamaEmbeddings
from langchain.schema import HumanMessage

from typing import Optional,List,Iterator,Any
import requests

import os
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = f"LLMDeepDiveChat"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_API_KEY"] = "ls__446246ca5f6744558aedc927aa852c05"  # Update with your API key


SALAD_API_KEY = "c598a341-4139-4e45-92e6-2880fbd61425"
SALAD_BASE_URL = 'https://sesame-panzanella-rzl32icxd728lczq.salad.cloud'

MODEL = "llama2"
class SaladChatOllama(ChatOllama):
    base_url: str = SALAD_BASE_URL

    def _create_stream(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> Iterator[str]:
        if self.stop is not None and stop is not None:
            raise ValueError("`stop` found in both the input and default params.")
        elif self.stop is not None:
            stop = self.stop
        elif stop is None:
            stop = []
        params = {**self._default_params, "stop": stop, **kwargs}
        response = requests.post(
            url=f"{self.base_url}/api/generate/",
            headers={"Content-Type": "application/json",
                     "Salad-Api-Key":SALAD_API_KEY},
            json={"prompt": prompt, **params},
            stream=True,
        )
        response.encoding = "utf-8"
        if response.status_code != 200:
            optional_detail = response.json().get("error")
            raise ValueError(
                f"Ollama call failed with status code {response.status_code}."
                f" Details: {optional_detail}"
            )
        return response.iter_lines(decode_unicode=True)
    

class SaladOllamaEmbeddings(OllamaEmbeddings):
    base_url: str = SALAD_BASE_URL

    def _process_emb_response(self, input: str) -> List[float]:
        """Process a response from the API.

        Args:
            response: The response from the API.

        Returns:
            The response as a dictionary.
        """
        headers={
            "Content-Type": "application/json",
            "Salad-Api-Key":SALAD_API_KEY
        }

        try:
            res = requests.post(
                f"{self.base_url}/api/embeddings",
                headers=headers,
                json={"model": self.model, "prompt": input, **self._default_params},
            )
        except requests.exceptions.RequestException as e:
            raise ValueError(f"Error raised by inference endpoint: {e}")

        if res.status_code != 200:
            raise ValueError(
                "Error raised by inference API HTTP code: %s, %s"
                % (res.status_code, res.text)
            )
        try:
            t = res.json()
            return t["embedding"]
        except requests.exceptions.JSONDecodeError as e:
            raise ValueError(
                f"Error raised by inference API: {e}.\nResponse: {res.text}"
            )
        

if __name__ == "__main__":
    # ollama = SaladChatOllama(base_url =SALAD_BASE_URL)
    # response = ollama([
    #                     HumanMessage(content="Tell me about the history of AI")
    #                 ])
    # print(response)

    ollama_emb = SaladOllamaEmbeddings()
    r1 = ollama_emb.embed_documents(
        [
            "Alpha is the first letter of Greek alphabet",
            "Beta is the second letter of Greek alphabet",
        ]
    )
    # r2 = ollama_emb.embed_query(
    #     "What is the second letter of Greek alphabet"
    # )
    print(r1)