from langchain_google_genai import ChatGoogleGenerativeAI# pyrefly: ignore [missing-import]
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate

load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

promt1 = PromptTemplate(
    template = "write a detailed report on the {topic}",
    input_variables=["topic"]
)

promt2 = PromptTemplate(
    template = "write a summary in 5 lines for following: \n {text}",
    input_variables=["text"]
)
parser = StrOutputParser()
chain = promt1 | model | parser | promt2 | model | parser

result = chain.invoke({"topic": "AI Engineer Role"})

print(result)

print(chain.get_graph().print_ascii())
