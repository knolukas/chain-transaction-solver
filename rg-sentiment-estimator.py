from datetime import datetime
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
    chainTransactionShare: float = Field(
        description="Percentage value of text that belongs to chain transaction description")
    chainTransactionDescription: str = Field(description="Text that describes the chain transaction")
    legalProcessShare: float = Field(description="Percentage value of text that belongs to legal process description")
    legalProcessDescription: str = Field(description="Text that describes the legal process")
    analysis: str = Field(description="Analysis of percentage estimation")


structured_llm = llm.with_structured_output(TextAnalysis)

parser = PydanticOutputParser(pydantic_object=TextAnalysis)
# Prompt
prompt_template = ChatPromptTemplate([
    ("system", "You are an expert in the legal field, you understand legal formulations and the technical jargon "
               "and you are an expert in the VAT area with regard to chain transactions in Austria and the "
               "European Union. Your job is to analyze a given text and estimate if this text is either a description "
               "of a"
               "chain transaction (category: Chain Transaction) or the description of a legal process (category: Legal Process)."
               "This is an example of a pure chain transaction description: \'Ein ungarischer Unternehmer U4 "
               "(=Empfänger) bestellt bei seinem italienischen Lieferanten U3 (=2. Erwerber) eine Maschine. "
               "Dieser wiederum bestellt die Maschine beim österreichischen Großhändler U2 (=1. Erwerber). "
               "Da der Großhändler U2 die Maschine nicht auf Lager hat, bestellt er diese beim deutschen Produzenten "
               "U1 (=Erstlieferant) und weist diesen an, die Maschine direkt an den ungarischen Unternehmer U4 zu liefern."
               "Wrap the output in 'json' tags\n{format_instructions}"
     ),
    ("human", "{text}"),
]).partial(format_instructions=parser.get_format_instructions())

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

prompt_template_extractor = ChatPromptTemplate([
    ("system", "You are a simple value extractor. In the given text there are percentage values for 2 categories."
               "You should only copy the relevant value. Do not change anything."
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


def extract_text(document_text, extract_type):
    if extract_type == "case":
        prompt = prompt_template_case.invoke(document_text)
        return llm.invoke(prompt).content

    elif extract_type == "verdict":
        prompt = prompt_template_verdict.invoke(document_text)
        return llm.invoke(prompt).content

    elif extract_type == "percentValueExtractionChain":
        prompt = prompt_template_extractor.invoke(
            {"output": "Print the number of the mentioned percentage share of category"
                       "Chain Transaction (format 80,0). Do not print anything else.",
             "text": document_text})
        return llm.invoke(prompt).content

    elif extract_type == "percentValueExtractionLegal":
        prompt = prompt_template_extractor.invoke(
            {"output": "Print the number of the mentioned percentage share of category"
                       "Legal process (format 80,0). Do not print anything else.",
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


def extract_text_from_pdf(pdf_path):
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text += page.extract_text()
    return text


# Pfad zum Projektverzeichnis
project_directory = "H:/Users/Lukas/OneDrive/Masterarbeit - LLMs in VAT - Knogler Lukas/Findok"

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

# Start LLM
df.loc[0:75, 'Response'] = df.loc[0:75, 'Case'].apply(lambda x: structured_llm.invoke(prompt_template.invoke(x)))
#
# # Verarbeite LLM Response
# df['Case'] = df['Response'].apply(lambda x: x.case)
# df['Verdict Description'] = df['Response'].apply(lambda x: x.verdict)
# df['Percentage Chain Transaction'] = df['Response'].apply(lambda x: x.chainTransactionShare)
# df['Chain Transaction Description'] = df['Response'].apply(lambda x: x.chainTransactionDescription)
# df['Percentage Legal Process'] = df['Response'].apply(lambda x: x.legalProcessShare)
# df['Percentage Legal Description'] = df['Response'].apply(lambda x: x.legalProcessDescription)
# df['Analysis'] = df['Response'].apply(lambda x: x.analysis)
df = pd.read_csv('dataframes/findok_data.csv')

df['Case'] = df['RawText'].apply(lambda x: extract_text(x, 'case'))
df.loc[26:, 'Analysis'] = df.loc[26:, 'Case'].apply(lambda x: extract_text(x, 'sentimentAnalysis'))
df.loc[0:75, '%ChainTransaction'] = df.loc[0:75, 'Analysis'].apply(
    lambda x: extract_text(x, 'percentValueExtractionChain'))
df.loc[0:75, '%LegalProcess'] = df.loc[0:75, 'Analysis'].apply(lambda x: extract_text(x, 'percentValueExtractionLegal'))

# df.loc[:10, 'Case'] = df.loc[:10, 'RawText'].apply(lambda x: extract_text(x, 'case'))

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
filename = f'output_{timestamp}.csv'
df.to_csv(os.path.join('dataframes', 'findok_data.csv'), index=False)
