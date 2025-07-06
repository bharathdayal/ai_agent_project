import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

# Load environment variables
load_dotenv()

# Initialize the ChatOpenAI instance
llm = ChatOpenAI(model="gpt-4o-mini")

# Take prompt as input from user
user_prompt = input("Enter your prompt: ")

# Invoke the model
response = llm.invoke(user_prompt)

# Print the result
print("Response:", response.content)