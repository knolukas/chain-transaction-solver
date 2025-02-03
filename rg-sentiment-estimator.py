import getpass
import os

import pdfplumber
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from langchain_neo4j import Neo4jGraph
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_openai import ChatOpenAI
from langchain_core.documents import Document
from dotenv import load_dotenv
from pydantic import BaseModel
import texts
import pandas as pd
from langchain_core.prompts import ChatPromptTemplate

prompt_template_sentiment = ChatPromptTemplate([
    ("system", "You are an expert in the legal field, you understand legal formulations and the technical jargon "
               "and you are an expert in the VAT area with regard to chain transactions in Austria and the "
               "European Union. Your job is to analyze a given text and estimate if this text is either a description "
               "of a"
               "chain transaction (category: Chain Transaction) or the description of a previous legal process or "
               "complaint regarding a chain transaction (category: Legal Process)."
               "terms like: Zollamt, Finanzamt, Bundesfinanzgericht, Bundesfinanzministerium, Zollanmeldung"
               "belong to category 'Legal Process'. "
               "Estimate the percentage of this text that fits into each category."
               "The output should be two percentage values that add up to exactly 100%."
               "{output}"),
    ("human", "{text}"),
])

prompt_template_case = ChatPromptTemplate([
    ("system", "Extract the whole section of this document where the case is described."
               "It should be marked as 'Sachverhalt' or 'Verfahrensgang' or related legal terms written in German language."
               "Do not change or add anything. Simply copy the relevant content."),
    ("human", "{text}"),
])

prompt_template_verdict = ChatPromptTemplate([
    ("system", "Extract the whole section of this document where the verdict of this case is described."
               "It should be marked as 'Erwaegungen' or 'Es wurde erwogen' or related legal terms written in German language."
               "do not change or add anything. Simply copy the relevant content."),
    ("human", "{text}"),
])

load_dotenv('.env')

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["NEO4J_URI"] = os.getenv("NEO4J_URI")
os.environ["NEO4J_USERNAME"] = os.getenv("NEO4J_USERNAME")
os.environ["NEO4J_PASSWORD"] = os.getenv("NEO4J_PASSWORD")
os.environ["LANGSMITH_TRACING"] = os.getenv("LANGSMITH_TRACING")
os.environ["LANGSMITH_ENDPOINT"] = os.getenv("LANGSMITH_ENDPOINT")
os.environ["LANGSMITH_API_KEY"] = os.getenv("LANGSMITH_API_KEY")
os.environ["LANGSMITH_PROJECT"] = os.getenv("LANGSMITH_PROJECT")

llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
)


# Pydantic
class TextAnalysis(BaseModel):
    """Text Analysis Structure"""
    case: str = Field(description="Copied case description")
    verdict: str = Field(description="Copied verdict description")
    chainTransactionShare: float = Field(description="Amount of text that belongs to chain transaction description")
    legalProcessShare: float = Field(description="Amount of text that belongs to legal process description")
    analysis: str = Field(description="Analysis of percentage estimation")


structured_llm = llm.with_structured_output(TextAnalysis)

parser = PydanticOutputParser(pydantic_object=TextAnalysis)
# Prompt
prompt_template = ChatPromptTemplate([
    ("system", "You are an expert in the legal field, you understand legal formulations and the technical jargon "
               "and you are an expert in the VAT area with regard to chain transactions in Austria and the "
               "European Union. Your job is to analyze a given text and estimate if this text is either a description "
               "of a"
               "chain transaction (category: Chain Transaction) or the description of a previous legal process or "
               "complaint regarding a chain transaction (category: Legal Process)."
               "Do not change or add anything. Simply copy the relevant content."
               "Wrap the output in 'json' tags\n{format_instructions}"
               "output node 'case': Extract the whole chapter of this document where the case is described."
               "The begin is marked as 'Sachverhalt' or 'Verfahrensgang' or related legal terms."
               " Do not change or add anything. Simply copy the relevant content."
               "output node 'verdict': Extract the whole chapter of this document where the verdict is described."
               "The begin is marked as 'Erwaegungen' or 'Es wurde erwogen' or related legal terms"
               "Do not change or add anything. Simply copy the relevant content."
               "output node 'chainTransactionShare': Analyze only the case description and estimate if this text is "
               "either a description of a"
               "chain transaction (category: Chain Transaction) or the description of a previous legal process or "
               "complaint regarding a chain transaction (category: Legal Process)."
               "terms like: Zollamt, Finanzamt, Bundesfinanzgericht, Bundesfinanzministerium, Zollanmeldung"
               "belong to category 'Legal Process'. "
               "Estimate the percentage of this text that fits into each category."
               "The output should be two percentage values that add up to exactly 100%. State the percentage value of "
               "'Chain Transaction' here."
               "output node 'legalProcessShare': state the percentage of 'Legal Process' that fits into"
               "output node 'analysis': Argue your estimation in textual form"
     ),
    ("human", "{text}"),
]).partial(format_instructions=parser.get_format_instructions())


# structured_llm.invoke("Tell me a joke about cats")

def extract_text_from_pdf(pdf_path):
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text += page.extract_text()
    return text


# Pfad zum Projektverzeichnis
project_directory = "input_data"

# Leere Liste für die extrahierten Daten
data = []

# Durchsuchen des Verzeichnisses nach PDF-Dateien
for filename in os.listdir(project_directory):
    if filename.endswith(".pdf"):
        pdf_path = os.path.join(project_directory, filename)
        raw_text = extract_text_from_pdf(pdf_path)

        data.append({"Filename": filename, "RawText": raw_text})

# Erstellen eines DataFrames
df = pd.DataFrame(data)

# Starte LLM
df['Response'] = df['RawText'].apply(lambda x: structured_llm.invoke(prompt_template.invoke(x)))

# Verarbeite LLM Response
df['Case'] = df['Response'].apply(lambda x: x.case)
df['Verdict Description'] = df['Response'].apply(lambda x: x.verdict)
df['Percentage Chain Transaction'] = df['Response'].apply(lambda x: x.chainTransactionShare)
df['Percentage Legal Process'] = df['Response'].apply(lambda x: x.legalProcessShare)
df['Analysis'] = df['Response'].apply(lambda x: x.analysis)


def extract_text(document_text, extract_type):
    if extract_type == "case":
        prompt = prompt_template_case.invoke(document_text)
        return llm.invoke(prompt).content

    elif extract_type == "verdict":
        prompt = prompt_template_verdict.invoke(document_text)
        return llm.invoke(prompt).content

    elif extract_type == "shareChain":
        prompt = prompt_template_sentiment.invoke({"output": "Only state the percentage value of "
                                                             "Chain Transaction as float number (0.5)",
                                                   "text": document_text})
        return llm.invoke(prompt).content

    elif extract_type == "shareLegal":
        prompt = prompt_template_sentiment.invoke({"output": "Only state the percentage value of "
                                                             "Legal Process as float number (0.5)",
                                                   "text": document_text})
        return llm.invoke(prompt).content

    elif extract_type == "sentimentAnalysis":
        prompt = prompt_template_sentiment.invoke({"output": "State the analysis and the whole description of your"
                                                             "estimation as text",
                                                   "text": document_text})
        return llm.invoke(prompt).content

    else:
        return "ERROR: Extract type must be 'case' or 'verdict' or 'sentiment'."