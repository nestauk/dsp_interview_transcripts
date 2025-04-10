import ast
import os

from typing import List
from typing import Tuple

from langchain.chains import LLMChain
from langchain.chat_models import AzureChatOpenAI
from langchain.prompts import PromptTemplate


llm = AzureChatOpenAI(
    openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    openai_api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    temperature=0,
)

answer_prompt = PromptTemplate.from_template(
    "Based on the following documents:\n\n{context}\n\n"
    "Answer the question: '{question}'\n"
    "Give a concise summary answer based only on the information provided."
)

quote_prompt = PromptTemplate.from_template(
    "Here are some documents:\n\n{context}\n\n"
    "The answer to the question '{question}' was: {answer}\n"
    "Extract the 3 documents that best support this answer."
    "Return them as a valid Python list of 3 strings. Do not modify the selected documents in any way. No explanation.\n"
    'Format: ["Doc 1", "Doc 2", "Doc 3"]'
)


def summarize_and_quote(texts: List[str], question: str) -> Tuple[str, List[str]]:
    """
    Generate a concise summary answer and extract supporting quotes for a research question
    based on a list of input texts.

    Args:
        texts (List[str]): A list of textual excerpts (e.g., sentences or paragraphs)
            extracted from transcripts.
        question (str): The research question to answer based on the input texts.

    Returns:
        Tuple[str, List[str]]:
            - A concise summary answer to the research question.
            - A list of 3 direct quotes from the input texts that support the answer.
              If parsing fails, a fallback list with a single "[Parsing error]" entry is returned.
    """
    context = "\n\n".join(texts)
    answer_chain = LLMChain(llm=llm, prompt=answer_prompt)
    answer = answer_chain.run(context=context, question=question)

    quote_chain = LLMChain(llm=llm, prompt=quote_prompt)
    quotes = quote_chain.run(context=context, question=question, answer=answer)

    try:
        quotes_list = ast.literal_eval(quotes)
    except Exception:
        quotes_list = ["[Parsing error]"]
    return answer, quotes_list
