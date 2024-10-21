import openai
from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()

temperatures = [0, 0.5, 1, 1.5, 2]

openai.api_key = os.getenv("CHATGPT_KEY")

client = OpenAI()

prompt = input("Enter the prompt: ")

for temperature in temperatures:
    completion = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        temperature=temperature
    )
    response_content = completion.choices[0].message.content.strip()
    print(f"\nTemperature: {temperature}")
    print(response_content)
    print("-" * 50)