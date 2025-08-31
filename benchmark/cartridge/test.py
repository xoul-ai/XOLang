"""
Quick demo of calling a local SGLang server (OpenAI-compatible API)
running on http://localhost:30003 with the official `openai` Python client.
"""

from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:30003/v1",
)

response = client.chat.completions.create(
    model="meta-llama/Llama-3.2-3B-Instruct",
    messages=[
        {
            "role": "user",
            "content": "Can you tell me about the patients?",
        },
    ],
)

print(response.choices[0].message.content)
