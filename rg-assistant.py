import pdfplumber
import openai
import os
import pandas as pd
from pydantic import BaseModel

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

# print(df)

# Speichern der Tabelle als CSV-Datei
# df.to_csv("extracted_data.csv", index=False)

client = openai.OpenAI(
    api_key="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"
)


# document_text = extract_text_from_pdf("input_data/116002.pdf")
# response = client.chat.completions.create(
#             model="gpt-4o-mini",  # Specify the correct model
#             messages=[
#                 {"role": "system", "content": "Calculate 1+1"},
#                 {"role": "user", "content": ""}
#             ],
#             temperature=0,
#             top_p=1
#         )


def extract_text(document_text, extract_type):
    prompt = "You are a simple legal text extractor."
    if extract_type not in ["case", "verdict"]:
        raise ValueError("Invalid Extract Type")

    if extract_type == "case":
        prompt += f"Extract the whole section of this document where the case is described. " \
                  f"It should be marked as 'Sachverhalt' or 'Verfahrensgang' or related legal terms written in German language. " \
                  f"Do not change or add anything. Simply copy the relevant content.\n\n"
    if extract_type == "verdict":
        print("enter verdict")
        prompt += f"Extract the whole section of this document where the verdict of this case is described. " \
                  f"It should be marked as 'Erwaegungen' or 'Es wurde erwogen' or related legal terms written in German language. " \
                  f"Do not change or add anything. Simply copy the relevant content.\n\n"

    try:
        print("enter llm")
        response = client.chat.completions.create(
            model="gpt-4o-mini",  # Specify the correct model
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": document_text}
            ],
            temperature=0,
            top_p=1
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error: {e}")
        return "Error occurred during extraction"


df['CaseDescription'] = df['RawText'].apply(lambda x: extract_text(x, "case"))
df['VerdictDescription'] = df['RawText'].apply(lambda x: extract_text(x, "verdict"))

# # Print the response from the assistant
# issue_content = response.choices[0].message.content
#
# assistant = client.beta.assistants.create(
#     name="RG Analyzer",
#     instructions="",
#     tools=[{'type': 'file_search'}],
#     model="gpt-4o-mini"
# )
#
# # Create a vector store
# vector_store = client.beta.vector_stores.create(name="chain transaction")
#
# # Read the files for upload to OpenAI
# file_paths = ["files/Das Reihengeschäft aus österreichischer Sicht.pdf",
#               "files/wko_reihengeschäft.pdf",
#               "files/Expose_Master_Thesis_Knogler.pdf"]
# file_streams = [open(path, "rb") for path in file_paths]
#
# # Use the upload and poll SDK helper to upload the files, add them to the vector store,
# # and poll the status of the file batch for completion.
# file_batch = client.beta.vector_stores.file_batches.upload_and_poll(
#     vector_store_id=vector_store.id, files=file_streams
# )
#
# # You can print the status and the file counts of the batch to see the result of this operation.
# print(file_batch.status)
# print(file_batch.file_counts)
#
# assistant = client.beta.assistants.update(
#     assistant_id=assistant.id,
#     tool_resources={"file_search": {"vector_store_ids": [vector_store.id]}},
# )
#
# # Add a new file to the vector store, this time in a PDF format. Note, you don't need to update the assistant again
# # as it's referring to the vector store id which has not changed.
# # file_02_path = "files/newFile.pdf"
# # file_02_stream = open(file_02_path, "rb")
# #
# # file_batch = client.beta.vector_stores.file_batches.upload_and_poll(
# #     vector_store_id=vector_store.id, files=[file_02_stream]
# # )
#
# # Create a thread
# thread = client.beta.threads.create()
#
# # Create a thread and attach the file to the message
# message_1 = client.beta.threads.messages.create(
#     thread_id=thread.id,
#     role="user",
#     content="Welche Forschungsfrage wird gestellt?",
# )
#
# # Create a run for the thread
# run = client.beta.threads.runs.create(
#     assistant_id=assistant.id,
#     thread_id=thread.id)
#
# run = client.beta.threads.runs.retrieve(
#     thread_id=thread.id,
#     run_id=run.id)
#
# messages = client.beta.threads.messages.list(
#     thread_id=thread.id,
#     order='asc'
# )
#
# for message in messages:
#     print(message.content[0].text.value)
#     print('\n')
