from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
load_dotenv()
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser
from langchain_core.runnables import RunnableBranch, RunnableLambda
from pydantic import BaseModel, Field

model = ChatGoogleGenerativeAI(model="gemma-4-26b-a4b-it")

parser = StrOutputParser()

from typing import Literal
class FeedBack(BaseModel):
    text: Literal['Positive', 'Negative'] = Field(description="Sentiment of the feedback")

parser2 = PydanticOutputParser(pydantic_object=FeedBack)

prompt1 = PromptTemplate(
    template="Tell the sentiment of the following review positive or negative: \n {review} \n {format_instruction}",
    input_variables=["review"],
    partial_variables={'format_instruction': parser2.get_format_instructions()}
)

prompt2 = PromptTemplate(
    template= "Write a appropriate response for the following Negaive review: \n {review}",
    input_variables=['review']
)
prompt3 = PromptTemplate(
    template= "Write a appropriate response for the following Positive review: \n {review}",
    input_variables=['review']
)

branch_chain = RunnableBranch(
    (lambda x: x.text == "Positive", prompt3 | model | parser),
    (lambda x: x.text == "Negative", prompt2 | model | parser),
    RunnableLambda(lambda x: "No sentimnet detected")
)

sentiment_chain = prompt1 | model | parser2
chain = sentiment_chain | branch_chain
result = chain.invoke({"review": "SAS Automation is a best Company"})
print(result)