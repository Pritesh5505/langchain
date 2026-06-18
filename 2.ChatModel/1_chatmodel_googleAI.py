from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

model = ChatGoogleGenerativeAI(model = "gemini-2.5-flash", temperature = 0.7, max_output_tokens = 30, thinking_budget=0)   # if not thinking_budget is set, the model will not print any content prior to the final output. this is specific to the chat model. 
result = model.invoke("write a fantasy story about a dragon and a princess in 5 lines")

print(result.content)