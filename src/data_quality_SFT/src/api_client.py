from openai import OpenAI
import tqdm
"""


"""
class llm_openai:
    def __init__(self, api_key, api_base, model_name) -> None:
        self.openai_api_key = api_key
        self.openai_api_base = api_base
        self.model_name = model_name
        self.client = OpenAI(
            api_key=api_key,
            base_url=api_base,
        )

    def chat(self, messages):
        # response = self.client.chat.completions.create(
        #     model=self.model_name,
        #     messages=messages,
        #     temperature=0.7,
        #     top_p=0.8,
        #     max_tokens=512,
        #     stream=False
        # )
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=0,
            stream=False
        )
        return response.choices[0].message.content
    
