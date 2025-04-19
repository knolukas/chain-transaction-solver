import itertools
import json
import os

import networkx as nx
import pandas as pd
from dotenv import load_dotenv
from langchain_community.graphs.graph_document import GraphDocument
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_experimental.graph_transformers.llm import LLMGraphTransformer
from langchain_neo4j import Neo4jGraph
from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import InMemoryVectorStore

from RAGModel import RAGModel

# Initialisiere einen globalen Counter für die Graph-IDs
graph_id_counter = itertools.count(start=20)


def get_next_graph_id():
    return f"graph_{next(graph_id_counter)}"


load_dotenv('.env')
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["NEO4J_URI"] = os.getenv("NEO4J_URI")
os.environ["NEO4J_USERNAME"] = os.getenv("NEO4J_USERNAME")
os.environ["NEO4J_PASSWORD"] = os.getenv("NEO4J_PASSWORD")

graph = Neo4jGraph(refresh_schema=False)

# Alle PDFs aus dem "files/" Ordner laden
pdf_files = [f for f in os.listdir("files") if f.endswith(".pdf")]

documents = []
for pdf in pdf_files:
    pdf_loader = PyPDFLoader(os.path.join("files", pdf))
    documents.extend(pdf_loader.load())

# Teile den Text in kleinere Abschnitte (Chunking)
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
docs = text_splitter.split_documents(documents)

vector_store = InMemoryVectorStore(OpenAIEmbeddings())
# Vektor-DB laden
vectorstore = FAISS.from_documents(docs, OpenAIEmbeddings())
vectorstore.save_local("files")

# FAISS-Datenbank laden (Sicherheitsoption beachten)
vectorstore = FAISS.load_local("files/", OpenAIEmbeddings(), allow_dangerous_deserialization=True)

# Retriever aus der Vektor-Datenbank erstellen
retriever = vectorstore.as_retriever()


def retrieve_knowledge(query, retriever):
    relevant_docs = retriever.get_relevant_documents(query)
    context = "\n".join([doc.page_content for doc in relevant_docs])
    return context


# LLM initialisieren
llm = ChatOpenAI(
    model="gpt-4o",
    temperature=0.2,
    max_tokens=None,
    timeout=None,
    max_retries=2,
)

rag_llm = RAGModel(retriever=retriever, llm=llm)


def delete_graph():
    query = "MATCH (n) DETACH DELETE n"
    graph.query(query)
    print("Graph deleted successfully")


def format_graph_documents(graph_docs):
    formatted_data = []

    for graph_doc in graph_docs:  # Iteriere über die Liste
        formatted_data.append({
            "nodes": [
                {"id": node.id, "type": node.type, "properties": node.properties}
                for node in graph_doc.nodes
            ],
            "relationships": [
                {
                    "source": rel.source.id,
                    "target": rel.target.id,
                    "type": rel.type,
                    "properties": rel.properties,
                }
                for rel in graph_doc.relationships
            ],
            "source_document": {
                "metadata": graph_doc.source.metadata,
                "content": graph_doc.source.page_content,
            },
        })

    return json.dumps(formatted_data, indent=4, ensure_ascii=False)


# async def process_graph():
#     data = await llm_transformer_props.aconvert_to_graph_documents(documents)
#     graph.add_graph_documents(data)
#     print(data)


def main():
    print("Hello, World!")


additional_instructions = ("system: Add the name of the transported good as "
                           "relationship property \"transported_good\".")

project_directory = "H:/Users/Lukas/OneDrive/Masterarbeit - LLMs in VAT - Knogler Lukas/"
df = pd.read_excel(project_directory + "Beispiele_Reihengeschäfte.xlsx")


def process_text_with_graph_transformer_static_03(text):
    allowed_nodes = ["Company", "Entrepreneur", "Carrier", "Transport"]
    allowed_relationships = [
        ("Company", "ORDERS_FROM", "Company"),
        ("Company", "ORDERS_FROM", "Entrepreneur"),
        ("Entrepreneur", "ORDERS_FROM", "Company"),
        ("Entrepreneur", "ORDERS_FROM", "Entrepreneur"),
        ("Company", "SELLS_TO", "Company"),
        ("Company", "SELLS_TO", "Entrepreneur"),
        ("Entrepreneur", "SELLS_TO", "Company"),
        ("Entrepreneur", "SELLS_TO", "Entrepreneur"),
        ("Company", "DELIVERS_TO", "Company"),
        ("Company", "DELIVERS_TO", "Entrepreneur"),
        ("Entrepreneur", "DELIVERS_TO", "Company"),
        ("Entrepreneur", "DELIVERS_TO", "Entrepreneur"),
        ("Carrier", "DELIVERS_TO", "Company"),
        ("Carrier", "DELIVERS_TO", "Entrepreneur"),
        ("Company", "COLLECTS_FROM", "Company"),
        ("Company", "COLLECTS_FROM", "Entrepreneur"),
        ("Entrepreneur", "COLLECTS_FROM", "Company"),
        ("Entrepreneur", "COLLECTS_FROM", "Entrepreneur"),
        ("Carrier", "COLLECTS_FROM", "Company"),
        ("Carrier", "COLLECTS_FROM", "Entrepreneur"),
        ("Company", "RESPONSIBLE_FOR", "Transport"),
        ("Entrepreneur", "RESPONSIBLE_FOR", "Transport"),
    ]
    node_properties = ["country", "UID", "name"]
    relationship_properties = ["date", "transported_good"]

    llm_transformer_props = LLMGraphTransformer(
        llm=llm,
        allowed_nodes=allowed_nodes,
        allowed_relationships=allowed_relationships,
        prompt=get_prompt(),
        strict_mode=False
    )

    docs = llm_transformer_props.convert_to_graph_documents([Document(page_content=text)])

    return docs


def process_text_with_graph_transformer_dynamic(text):
    allowed_nodes = ["Company", "Entrepreneur", "Carrier", "Transport"]

    relationships = ["ORDERS_FROM", "DELIVERS_TO", "COLLECTS_FROM"]
    transport_relationship = "RESPONSIBLE_FOR"

    allowed_relationships = [
        (start, rel, end)
        for start in allowed_nodes for end in allowed_nodes for rel in relationships
    ]

    allowed_relationships += [
        (start, transport_relationship, "Transport")
        for start in ["Company", "Entrepreneur"]
        if "Transport" in allowed_nodes
    ]

    node_properties = ["country", "UID", "name"]
    relationship_properties = ["date", "transported_good"]

    llm_transformer_props = LLMGraphTransformer(
        llm=llm,
        allowed_nodes=allowed_nodes,
        allowed_relationships=allowed_relationships,
        prompt=get_prompt(),
        strict_mode=False
    )

    docs = llm_transformer_props.convert_to_graph_documents([Document(page_content=text)])

    return docs


def process_text_with_graph_transformer_standard_v1(text):
    allowed_nodes = ["Company", "Entrepreneur", "Carrier", "Transport"]
    allowed_relationships = [
        ("Company", "ORDERS_FROM", "Company"),
        ("Company", "ORDERS_FROM", "Entrepreneur"),
        ("Entrepreneur", "ORDERS_FROM", "Company"),
        ("Entrepreneur", "ORDERS_FROM", "Entrepreneur"),
        ("Company", "SELLS_TO", "Company"),
        ("Company", "SELLS_TO", "Entrepreneur"),
        ("Entrepreneur", "SELLS_TO", "Company"),
        ("Entrepreneur", "SELLS_TO", "Entrepreneur"),
        ("Company", "DELIVERS_TO", "Company"),
        ("Company", "DELIVERS_TO", "Entrepreneur"),
        ("Entrepreneur", "DELIVERS_TO", "Company"),
        ("Entrepreneur", "DELIVERS_TO", "Entrepreneur"),
        ("Carrier", "DELIVERS_TO", "Company"),
        ("Carrier", "DELIVERS_TO", "Entrepreneur"),
        ("Company", "COLLECTS_FROM", "Company"),
        ("Company", "COLLECTS_FROM", "Entrepreneur"),
        ("Entrepreneur", "COLLECTS_FROM", "Company"),
        ("Entrepreneur", "COLLECTS_FROM", "Entrepreneur"),
        ("Carrier", "COLLECTS_FROM", "Company"),
        ("Carrier", "COLLECTS_FROM", "Entrepreneur"),
        ("Company", "RESPONSIBLE_FOR", "Transport"),
        ("Entrepreneur", "RESPONSIBLE_FOR", "Transport"),
    ]
    node_properties = ["country", "UID", "name"]
    relationship_properties = ["date", "transported_good"]

    llm_transformer_props = LLMGraphTransformer(
        llm=llm,
        allowed_nodes=allowed_nodes,
        allowed_relationships=allowed_relationships,
    )

    docs = llm_transformer_props.convert_to_graph_documents([Document(page_content=text)])

    return docs


def process_text_with_graph_transformer_standard_v2(text):
    allowed_nodes = ["Company", "Entrepreneur", "Carrier", "Transport"]
    allowed_relationships = [
        ("Company", "ORDERS_FROM", "Company"),
        ("Company", "ORDERS_FROM", "Entrepreneur"),
        ("Entrepreneur", "ORDERS_FROM", "Company"),
        ("Entrepreneur", "ORDERS_FROM", "Entrepreneur"),
        ("Company", "SELLS_TO", "Company"),
        ("Company", "SELLS_TO", "Entrepreneur"),
        ("Entrepreneur", "SELLS_TO", "Company"),
        ("Entrepreneur", "SELLS_TO", "Entrepreneur"),
        ("Company", "DELIVERS_TO", "Company"),
        ("Company", "DELIVERS_TO", "Entrepreneur"),
        ("Entrepreneur", "DELIVERS_TO", "Company"),
        ("Entrepreneur", "DELIVERS_TO", "Entrepreneur"),
        ("Carrier", "DELIVERS_TO", "Company"),
        ("Carrier", "DELIVERS_TO", "Entrepreneur"),
        ("Company", "COLLECTS_FROM", "Company"),
        ("Company", "COLLECTS_FROM", "Entrepreneur"),
        ("Entrepreneur", "COLLECTS_FROM", "Company"),
        ("Entrepreneur", "COLLECTS_FROM", "Entrepreneur"),
        ("Carrier", "COLLECTS_FROM", "Company"),
        ("Carrier", "COLLECTS_FROM", "Entrepreneur"),
        ("Company", "RESPONSIBLE_FOR", "Transport"),
        ("Entrepreneur", "RESPONSIBLE_FOR", "Transport"),
        ("Company", "RESPONSIBLE_FOR", "Carrier"),
        ("Entrepreneur", "RESPONSIBLE_FOR", "Carrier"),
    ]
    node_properties = ["country", "UID", "name"]
    relationship_properties = ["date", "transported_good"]

    llm_transformer_props = LLMGraphTransformer(
        llm=llm,
        allowed_nodes=allowed_nodes,
        allowed_relationships=allowed_relationships,
        strict_mode=False,
    )

    docs = llm_transformer_props.convert_to_graph_documents([Document(page_content=text)])

    return docs


def process_text_with_graph_transformer_standard_v3(text):
    allowed_nodes = ["Company", "Entrepreneur", "Carrier", "Transport"]
    allowed_relationships = [
        ("Company", "ORDERS_FROM", "Company"),
        ("Company", "ORDERS_FROM", "Entrepreneur"),
        ("Entrepreneur", "ORDERS_FROM", "Company"),
        ("Entrepreneur", "ORDERS_FROM", "Entrepreneur"),
        ("Company", "SELLS_TO", "Company"),
        ("Company", "SELLS_TO", "Entrepreneur"),
        ("Entrepreneur", "SELLS_TO", "Company"),
        ("Entrepreneur", "SELLS_TO", "Entrepreneur"),
        ("Company", "DELIVERS_TO", "Company"),
        ("Company", "DELIVERS_TO", "Entrepreneur"),
        ("Entrepreneur", "DELIVERS_TO", "Company"),
        ("Entrepreneur", "DELIVERS_TO", "Entrepreneur"),
        ("Carrier", "DELIVERS_TO", "Company"),
        ("Carrier", "DELIVERS_TO", "Entrepreneur"),
        ("Company", "COLLECTS_FROM", "Company"),
        ("Company", "COLLECTS_FROM", "Entrepreneur"),
        ("Entrepreneur", "COLLECTS_FROM", "Company"),
        ("Entrepreneur", "COLLECTS_FROM", "Entrepreneur"),
        ("Carrier", "COLLECTS_FROM", "Company"),
        ("Carrier", "COLLECTS_FROM", "Entrepreneur"),
        ("Company", "RESPONSIBLE_FOR", "Transport"),
        ("Entrepreneur", "RESPONSIBLE_FOR", "Transport"),
        ("Company", "RESPONSIBLE_FOR", "Carrier"),
        ("Entrepreneur", "RESPONSIBLE_FOR", "Carrier"),
    ]
    node_properties = ["country", "UID"]
    relationship_properties = ["date", "transported_good"]

    llm_transformer_props = LLMGraphTransformer(
        llm=llm,
        allowed_nodes=allowed_nodes,
        allowed_relationships=allowed_relationships,
    )

    docs = llm_transformer_props.convert_to_graph_documents([Document(page_content=text)])

    return docs


def process_text_with_graph_transformer_standard_v4(text):
    allowed_nodes = ["Company", "Entrepreneur", "Carrier", "Transport"]
    allowed_relationships = [
        ("Company", "ORDERS_FROM", "Company"),
        ("Company", "ORDERS_FROM", "Entrepreneur"),
        ("Entrepreneur", "ORDERS_FROM", "Company"),
        ("Entrepreneur", "ORDERS_FROM", "Entrepreneur"),
        ("Company", "SELLS_TO", "Company"),
        ("Company", "SELLS_TO", "Entrepreneur"),
        ("Entrepreneur", "SELLS_TO", "Company"),
        ("Entrepreneur", "SELLS_TO", "Entrepreneur"),
        ("Company", "DELIVERS_TO", "Company"),
        ("Company", "DELIVERS_TO", "Entrepreneur"),
        ("Entrepreneur", "DELIVERS_TO", "Company"),
        ("Entrepreneur", "DELIVERS_TO", "Entrepreneur"),
        ("Carrier", "DELIVERS_TO", "Company"),
        ("Carrier", "DELIVERS_TO", "Entrepreneur"),
        ("Company", "COLLECTS_FROM", "Company"),
        ("Company", "COLLECTS_FROM", "Entrepreneur"),
        ("Entrepreneur", "COLLECTS_FROM", "Company"),
        ("Entrepreneur", "COLLECTS_FROM", "Entrepreneur"),
        ("Carrier", "COLLECTS_FROM", "Company"),
        ("Carrier", "COLLECTS_FROM", "Entrepreneur"),
        ("Company", "RESPONSIBLE_FOR", "Transport"),
        ("Entrepreneur", "RESPONSIBLE_FOR", "Transport"),
        ("Company", "RESPONSIBLE_FOR", "Carrier"),
        ("Entrepreneur", "RESPONSIBLE_FOR", "Carrier"),
    ]
    node_properties = ["country", "UID"]
    relationship_properties = ["date", "transported_good"]

    llm_transformer_props = LLMGraphTransformer(
        llm=rag_llm,
        allowed_nodes=allowed_nodes,
        allowed_relationships=allowed_relationships,
        node_properties=node_properties,
        relationship_properties=relationship_properties,
    )

    docs = llm_transformer_props.convert_to_graph_documents([Document(page_content=text)])

    return docs


def process_text_with_graph_transformer_standard_v5(text):
    allowed_nodes = ["Company", "Entrepreneur", "Carrier", "Transport"]
    allowed_relationships = [
        ("Company", "ORDERS_FROM", "Company"),
        ("Company", "ORDERS_FROM", "Entrepreneur"),
        ("Entrepreneur", "ORDERS_FROM", "Company"),
        ("Entrepreneur", "ORDERS_FROM", "Entrepreneur"),
        ("Company", "SELLS_TO", "Company"),
        ("Company", "SELLS_TO", "Entrepreneur"),
        ("Entrepreneur", "SELLS_TO", "Company"),
        ("Entrepreneur", "SELLS_TO", "Entrepreneur"),
        ("Company", "DELIVERS_TO", "Company"),
        ("Company", "DELIVERS_TO", "Entrepreneur"),
        ("Entrepreneur", "DELIVERS_TO", "Company"),
        ("Entrepreneur", "DELIVERS_TO", "Entrepreneur"),
        ("Carrier", "DELIVERS_TO", "Company"),
        ("Carrier", "DELIVERS_TO", "Entrepreneur"),
        ("Company", "COLLECTS_FROM", "Company"),
        ("Company", "COLLECTS_FROM", "Entrepreneur"),
        ("Entrepreneur", "COLLECTS_FROM", "Company"),
        ("Entrepreneur", "COLLECTS_FROM", "Entrepreneur"),
        ("Carrier", "COLLECTS_FROM", "Company"),
        ("Carrier", "COLLECTS_FROM", "Entrepreneur"),
        ("Company", "RESPONSIBLE_FOR", "Transport"),
        ("Entrepreneur", "RESPONSIBLE_FOR", "Transport"),
        ("Company", "RESPONSIBLE_FOR", "Carrier"),
        ("Entrepreneur", "RESPONSIBLE_FOR", "Carrier"),
    ]
    node_properties = ["country", "UID"]
    relationship_properties = ["date", "transported_good"]
    global retriever
    context = retrieve_knowledge(text, retriever)

    llm_transformer_props = LLMGraphTransformer(
        llm=llm,
        allowed_nodes=allowed_nodes,
        allowed_relationships=allowed_relationships,
        node_properties=node_properties,
        relationship_properties=relationship_properties,
        additional_instructions="Use this as additional knowledge-base:" + context
    )

    docs = llm_transformer_props.convert_to_graph_documents([Document(page_content=text)])

    return docs


def process_text_with_prompt(text):
    prompt_template = ChatPromptTemplate([
        ("system",
         "Extract from the given text a JSON_LD formatted Knowledge Graph. th graph should include all entities and relationsships with "
         "corresponding properties according to the following schema. Do not use any extra keys or fields or properties."
         """
                    allowed_nodes = "Company", "Entrepreneur", "Carrier", "Transport"
                    allowed_relationships = [
                        ("Company", "ORDERS_FROM", "Company"),
                        ("Company", "ORDERS_FROM", "Entrepreneur"),
                        ("Entrepreneur", "ORDERS_FROM", "Company"),
                        ("Entrepreneur", "ORDERS_FROM", "Entrepreneur"),
                        ("Company", "SELLS_TO", "Company"),
                        ("Company", "SELLS_TO", "Entrepreneur"),
                        ("Entrepreneur", "SELLS_TO", "Company"),
                        ("Entrepreneur", "SELLS_TO", "Entrepreneur"),
                        ("Company", "DELIVERS_TO", "Company"),
                        ("Company", "DELIVERS_TO", "Entrepreneur"),
                        ("Entrepreneur", "DELIVERS_TO", "Company"),
                        ("Entrepreneur", "DELIVERS_TO", "Entrepreneur"),
                        ("Carrier", "DELIVERS_TO", "Company"),
                        ("Carrier", "DELIVERS_TO", "Entrepreneur"),
                        ("Company", "COLLECTS_FROM", "Company"),
                        ("Company", "COLLECTS_FROM", "Entrepreneur"),
                        ("Entrepreneur", "COLLECTS_FROM", "Company"),
                        ("Entrepreneur", "COLLECTS_FROM", "Entrepreneur"),
                        ("Carrier", "COLLECTS_FROM", "Company"),
                        ("Carrier", "COLLECTS_FROM", "Entrepreneur"),
                        ("Company", "RESPONSIBLE_FOR", "Transport"),
                        ("Entrepreneur", "RESPONSIBLE_FOR", "Transport"),
                        ("Company", "RESPONSIBLE_FOR", "Carrier"),
                        ("Entrepreneur", "RESPONSIBLE_FOR", "Carrier"),
                    ]
                    node_properties = ["name", "country", "VAT number"]
                    relationship_properties = ["date", "transported_good"]"""
         ),
        ("human", "{text}"),
    ])

    prompt = prompt_template.invoke(text)
    return llm.invoke(prompt).content


# probably useful rule:
#
# "## 4. Completeness\n"
#         "Each chain transaction consists of at least these nodes and relations:\n"
#         '- At least 3 or more different companies and/or entrepreneurs\n'
#         '- Make sure that every node has at least one relationship.\n'
#         '- At least one INITIATES or COLLECTS_FROM or DELIVERS_TO relationship.'


def get_prompt(
        additional_instructions: str = "",
) -> ChatPromptTemplate:
    return ChatPromptTemplate.from_messages(
        [
            ("system", open('system_prompt_03.txt', encoding='utf-8').read()),
            (
                "human",
                additional_instructions
                + " Tip: Make sure to answer in the correct format and do "
                  "not include any explanations. "
                  "Use the given format to extract information from the "
                  "following input: {input}",
            ),
        ]
    )


def process_text_with_graph_transformer_v2(text):
    """Wendet den LLM-Graph-Transformer an und fügt die Graph-ID hinzu."""
    graph_id = get_next_graph_id()  # Nächste Graph-ID holen
    allowed_nodes = ["Company", "Delivery", "Invoice", "Order", "Product", "Country", "Carrier"]
    allowed_relationships = [
        ("Company", "ORDERS", "Company"),
        ("Delivery", "INITIATED_BY", "Company"),
        ("Delivery", "TRANSPORTED_BY", "Company"),
        ("Delivery", "TRANSPORTED_BY", "Carrier"),
        ("Company", "LOCATED_IN", "Country"),
        ("Product", "PART_OF", "Delivery"),
    ]
    node_properties = ["country", "UID", "name"]
    relationship_properties = ["date", "transported_good"]
    llm_transformer_props = LLMGraphTransformer(
        llm=llm,
        allowed_nodes=allowed_nodes,
        allowed_relationships=allowed_relationships,
        prompt=get_prompt()
    )

    docs = llm_transformer_props.process_response(Document(page_content=text))

    return docs


output = process_text_with_graph_transformer_standard_v4(df.loc[18, 'text'])  # für kleine tests
# Wende die Funktion auf die DataFrame-Spalte an
# df.loc[0:2, 'graph'] = df.loc[0:2, 'text'].apply(process_text_with_graph_transformer)
# df.loc[0:5, 'graph2'] = df.loc[0:5, 'text'].apply(process_text_with_graph_transformer_v2)  # standard prompt
# df.loc[0:0, 'graph1'] = df.loc[0:0, 'text'].apply(lambda x: process_text_with_graph_transformer_v2(x))  # hier mit neuer prompt
# df.loc[0:4, 'graph0'] = df.loc[0:4, 'text'].apply(lambda x: process_text_with_graph_transformer_generic(x))

df['graph_data_test'] = df['text'].apply(lambda x: process_text_with_prompt(x))

df.loc[0:10, 'graph_data_test'] = df.loc[0:10, 'text'].apply(
    lambda x: process_text_with_prompt(x))
# test code snipped
delete_graph()
data = df.at[21, 'graph_data_10']

df_experiment = pd.read_csv('')


def restructure_json_graph(json_data):
    data = json.loads(json_data.removesuffix("```").strip().split('\n', 1)[1])

    # Graph erstellen
    G = nx.DiGraph()

    # Knoten hinzufügen
    for node in data["@graph"]:
        UID = node["UID"]
        G.add_node(UID, type=node["@type"], **{k: v for k, v in node.items() if k not in ["@type", "UID"]})

    # Kanten hinzufügen
    for node in data["@graph"]:
        UID = node["UID"]
        for rel, target in node.items():
            if isinstance(target, dict) and "@type" in target and "UID" in target:
                G.add_edge(UID, target["UID"], relationship=rel)

    # Graph-Infos ausgeben
    print("Knoten:", G.nodes(data=True))
    print("Kanten:", G.edges(data=True))

df_langsmith = pd.read_csv("H:/Users/Lukas/Downloads/pg__v_002__gpt-4o__e95af38f (1).csv")

data = json.loads(df_langsmith.at[0, 'run'])
data_2 = data['outputs']['output']['content']


def display_graph(row):
    data = json.loads(row)
    print(type(data))
    graph_data = data['outputs']['output']['content']
    print((graph_data))
    if isinstance(graph_data, str):
        graph_data = json.loads(graph_data)
    print(type(graph_data))
    # Nodes anzeigen
    print("Nodes:")
    for node in graph_data['nodes']:
        print(f"ID: {node['id']}, Type: {node['type']}")
        for key, value in node["properties"].items():
            print(f"  {key}: {value}")
        print()

    # Relationships anzeigen
    print("Graph with properties:")
    rel_text_w_prop = ", ".join(
        f"{relationship['source']} -- [{relationship['type']} "
        + ", ".join(f"{key}: {value}" for key, value in relationship["properties"].items())
        + f"] -> {relationship['target']}"
        for relationship in graph_data["relationships"]
    )
    print(rel_text_w_prop)

    print("Graph w/o properties:")
    rel_text_wo_prop = ", ".join(
        f"{relationship['source']} -- {relationship['type']} -> {relationship['target']} "
        for relationship in graph_data["relationships"]
    )

    print(rel_text_wo_prop)

    return rel_text_wo_prop

#anwenden auf ganzen dataframe
df_langsmith['graph_data_text'] = df_langsmith['run'].apply(display_graph)


# Now you can access the 'nodes' field
index = '10'
df_name = "graph_data_" + index
df_name = "outputs"
range = df.index

for idx in range:
    # if pd.isna(df.at[idx, df_name]):  # Prüfen, ob df_name NaN ist
    #     df.at[idx, 'nodes_' + index] = "NaN"
    #     df.at[idx, 'relationships_' + index] = "NaN"
    #     continue

    nodes = df.at[idx, df_name]['output']['content'].nodes
    relationships = df.at[idx, df_name]['output']['content'].relationships

    # Formatierte Ausgabe der Knoten
    node_text = ", ".join(
        f"{node.id} (Typ: {node.type}" + (f", Properties: {node.properties}" if node.properties else "") + ")"
        for node in nodes
    )

    # Formatierte Ausgabe der Beziehungen
    rel_text = ", ".join(
        f"{rel.source.id} --[{rel.type}" + (
            f" (Properties: {rel.properties})" if rel.properties else "") + f"]--> {rel.target.id}"
        for rel in relationships
    )

    # Speichern der Textrepräsentation im DataFrame
    df.at[idx, 'nodes_' + index] = node_text
    df.at[idx, 'relationships_' + index] = rel_text

df.to_excel('dataframes/df_edited_graph_data.xlsx', index=True)

find_final_receiver = """OPTIONAL MATCH (n) 
    WHERE NOT n:Transport AND NOT EXISTS { MATCH ()-[:ORDERS_FROM]->(n) }

    OPTIONAL MATCH (m) 
    WHERE NOT m:Transport AND NOT EXISTS { MATCH (m)-[:SELLS_TO]->() }

    WITH n,m
    WHERE n IS NOT NULL OR m IS NOT NULL

    RETURN
      CASE 
        WHEN n <> m THEN "Inconsistent, no solution."
        WHEN n IS NOT NULL THEN n
        WHEN m IS NOT NULL THEN m
        ELSE "Inconsistent, no solution."
      END AS result"""
find_supplier = """OPTIONAL MATCH (n) 
    WHERE NOT n:Transport AND NOT EXISTS { MATCH (n)-[:ORDERS_FROM]->() }

OPTIONAL MATCH (m) 
    WHERE NOT m:Transport AND NOT EXISTS { MATCH ()-[:SELLS_TO]->(m) }

WITH n,m
    WHERE n IS NOT NULL OR m IS NOT NULL

    RETURN
      CASE 
        WHEN n <> m THEN "Inconsistent, no solution."
        WHEN n IS NOT NULL THEN n
        WHEN m IS NOT NULL THEN m
        ELSE "Inconsistent, no solution."
      END AS result"""

id_value = "Ch"
find_first_revenue = f"""OPTIONAL MATCH (n {{id: "{id_value}"}})-[r:ORDERS_FROM]-(m)
OPTIONAL MATCH (n)-[r:SELLS_TO]-(m {{id: "{id_value}"}})
RETURN n, r, m"""
find_last_revenue = f"""OPTIONAL MATCH (n)-[r:ORDERS_FROM]-(m {{id: "{id_value}"}})
OPTIONAL MATCH (n {{id: "{id_value}"}})-[r:SELLS_TO]-(m)
RETURN n, r, m"""

# Füge die generierten Graphen zu `graph` hinzu (in die Neo4j Datenbank)
id_counter = itertools.count(start=0)


def add_graph():
    delete_graph()
    data = df.at[next(id_counter), 'graph_data_10']
    print(data)
    graph.add_graph_documents(data)


def add_graph_and_query(query):
    add_graph()
    results = graph.query(
        query
    )
    for result in results:
        print(result['result'])


add_graph_and_query(find_final_receiver)
add_graph_and_query(find_supplier)
add_graph_and_query(find_first_revenue)
add_graph_and_query(find_last_revenue)

# ******************************************
# Textuelle Visualisierung im DataFrame


# additional_instructions_good = ("system: First, identify the good of interest and then interpret the whole transaction"
#                                 "as a pov from the good.")
# documents = [Document(page_content=reihegeschaeft_bsp + additional_instructions)]
# data = llm_transformer_props.convert_to_graph_documents(documents)


# Run the async function in the event loop
# asyncio.run(process_graph())
# # Print GraphDocuments
# print(format_graph_documents(data))

if __name__ == "__main__":
    main()

# llm_transformer_tuple = LLMGraphTransformer(
#     llm=llm,
#     allowed_nodes=["Person", "Country", "Organization"],
#     allowed_relationships=allowed_relationships,
# )
graph_documents_filtered = llm_transformer_props.convert_to_graph_documents(documents)
print(f"Nodes:{graph_documents_filtered[0].nodes}")
print(f"Relationships:{graph_documents_filtered[0].relationships}")

graph.add_graph_documents(graph_documents_filtered)

# """
# Third approach with additional node properties
# """
# llm_transformer_props = LLMGraphTransformer(
#     llm=llm,
#     allowed_nodes=["Person", "Country", "Organization"],
#     allowed_relationships=["NATIONALITY", "LOCATED_IN", "WORKED_AT", "SPOUSE"],
#     node_properties=["born_year"],
# )
# graph_documents_props = llm_transformer_props.convert_to_graph_documents(documents)
# print(f"Nodes:{graph_documents_props[0].nodes}")
# print(f"Relationships:{graph_documents_props[0].relationships}")
#
# graph.add_graph_documents(graph_documents_props)
