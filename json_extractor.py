import datetime
import json
import os

import pandas as pd
from dotenv import load_dotenv
from langchain_neo4j import Neo4jGraph
import datetime
import networkx as nx
import matplotlib.pyplot as plt


#def main():
load_dotenv('.env')
os.environ["NEO4J_URI"] = os.getenv("NEO4J_URI")
os.environ["NEO4J_USERNAME"] = os.getenv("NEO4J_USERNAME")
os.environ["NEO4J_PASSWORD"] = os.getenv("NEO4J_PASSWORD")

def delete_graph():
    query = "MATCH (n) DETACH DELETE n"
    graph.query(query)
    print("Graph deleted successfully")

def build_graph(query):
    data = graph.query(query)
    print("Graph built successfully")
    return data


filename = "pg__voo5__gpt-4.1__0f7491aa.csv"
df_langsmith = pd.read_csv("H:/Users/Lukas/Downloads/" + filename)
internal_ids = [14, 9, 31, 13, 7, 6, 4, 1, 10, 35, 17, 30, 27, 5, 18, 32, 34, 2, 36, 20, 29, 28, 37, 38, 3, 39, 19, 11,
                8, 15, 25, 33, 26, 24, 23, 12, 16, 22, 21]
quelle = [
    "KOLLMANN", "KOLLMANN", "KOLLMANN", "KOLLMANN", "KOLLMANN",
    "KOLLMANN", "KOLLMANN", "KOLLMANN", "KOLLMANN", "KOLLMANN",
    "KOLLMANN", "KOLLMANN", "KOLLMANN", "KOLLMANN", "KOLLMANN",
    "KOLLMANN", "KOLLMANN", "KOLLMANN", "KOLLMANN", "KOLLMANN",
    "REALCASES", "REALCASES", "REALCASES", "REALCASES", "REALCASES",
    "REALCASES", "REALCASES", "REALCASES", "REALCASES", "REALCASES",
    "REALCASES", "REALCASES", "REALCASES", "REALCASES", "REALCASES",
    "REALCASES", "REALCASES", "REALCASES", "REALCASES"
]
df_langsmith.insert(loc=0,column='interne_ID',value=internal_ids)
df_langsmith.insert(loc=1,column='quelle',value=quelle)
graph = Neo4jGraph(refresh_schema=False)


query_endkunde = """
OPTIONAL MATCH (n) 
WHERE NOT n:Transportverantwortung AND NOT EXISTS { MATCH ()-[:BESTELLUNG]->(n) } 

RETURN COALESCE(n, "Inconsistent, no solution.") AS result
"""

query_hersteller = """
OPTIONAL MATCH (n) 
WHERE NOT n:Transportverantwortung AND NOT EXISTS { MATCH (n)-[:BESTELLUNG]->() } 

RETURN COALESCE(n, "Inconsistent, no solution.") AS result"""

query_finde_transport = """
    //Finde Lieferung
    OPTIONAL MATCH (n:Partei)-[r:WARENBEWEGUNG]->(m:Partei) 
    RETURN n, r, m
    """
query_reihengeschäft_prüfen = """
    MATCH path = (start:Unternehmen)-[r1:BESTELLUNG*]->(end:Unternehmen)
    OPTIONAL MATCH delivery_path = (n)-[r2:WARENBEWEGUNG]->(start) 
    WHERE size(r1) >= 2  // Mindestens zwei BESTELLUNGEN müssen existieren
    WITH path, delivery_path, r1, r2, start, end
    RETURN 
        CASE 
            WHEN r2 IS NULL THEN "Kein Reihengeschäft: Keine direkte Lieferung vom letzten zum ersten Käufer."
            WHEN size(r1) < 2 THEN "Kein Reihengeschäft: Nur 2 Unternehmen involviert."
            WHEN NOT ALL(rel IN relationships(path) WHERE rel.Produkt = r2.Produkt) 
                THEN "Kein Reihengeschäft: Produktabweichung in der Bestellkette."
            ELSE 
                {Beteiligte: nodes(path), 
                  Bestellungen: (path), 
                  Lieferung: delivery_path, 
                  Status: "Reihengeschäft erkannt!" }
        END AS Ergebnis"""

query_transportverantwortung = """
    MATCH (n:Unternehmen)-[:HAT]->(:Transportverantwortung)
    RETURN n"""

full_df = df_langsmith.index
selection = range(12, 13)


for idx in full_df:
    try:
        #idx = 12
        delete_graph()
        df_langsmith.at[idx, 'outputs_decoded'] = df_langsmith.at[idx, 'outputs'].encode().decode('unicode_escape')
        df_langsmith.at[idx, 'inputs_decoded'] = df_langsmith.at[idx, 'inputs'].encode().decode('unicode_escape')
        print(f"start with id: {idx} \n")
        try:
            data = json.loads(df_langsmith.at[idx, 'outputs'])
            #print(data)
        except json.JSONDecodeError as e:
            print(f"JSONDecodeError: {e}")

        cypher_statement = data['Cypher Anweisungen']
        cypher_statement = cypher_statement.replace("\\n", "\n")
        cypher_statement = cypher_statement.replace(";", "")
        cypher_statement = cypher_statement.replace("name", "Name")
        cypher_statement = cypher_statement.replace("sitz", "Sitz")
        cypher_statement = cypher_statement.replace("uid", "UID")
        #print(cypher_statement)
        df_langsmith.at[idx, 'cypher_statements'] = cypher_statement
        graph_data = build_graph(cypher_statement)

        hersteller = graph.query(query_hersteller)
        df_langsmith.at[idx, 'erstes Unternehmen'] = str(hersteller)
        df_langsmith.at[idx, 'erstes Unternehmen'] = df_langsmith.at[idx, 'erstes Unternehmen'].replace("'", '"')
        hersteller_data = json.loads(df_langsmith.at[idx, 'erstes Unternehmen'])
        hersteller_name = hersteller_data[0]['result']['Name']

        endkunde = graph.query(query_endkunde)
        df_langsmith.at[idx, 'letztes Unternehmen'] = str(endkunde)
        df_langsmith.at[idx, 'letztes Unternehmen'] = df_langsmith.at[idx, 'letztes Unternehmen'].replace("'", '"')
        endkunde_data = json.loads(df_langsmith.at[idx, 'letztes Unternehmen'])
        endkunde_name = endkunde_data[0]['result']['Name']

        query_erster_umsatz = f"""
           //variable für Name "letztes Unternehmen"
           OPTIONAL MATCH (n:Unternehmen {{Name: "{endkunde_name}"}})-[r:BESTELLUNG]-(m:Unternehmen)
           RETURN n, r, m"""
        erster_umsatz = graph.query(query_erster_umsatz)
        df_langsmith.at[idx, 'erster_umsatz'] = str(erster_umsatz)
        df_langsmith.at[idx, 'erster_umsatz'] = df_langsmith.at[idx, 'erster_umsatz'].replace("'", '"')

        query_letzter_umsatz = f"""
            //variable für Name "erstes Unternehme"
            OPTIONAL MATCH (n:Unternehmen)-[r:BESTELLUNG]-(m:Unternehmen {{Name: "{hersteller_name}"}})
            RETURN n, r, m"""
        letzter_umsatz = graph.query(query_letzter_umsatz)
        df_langsmith.at[idx, 'letzter_umsatz'] = str(letzter_umsatz)
        df_langsmith.at[idx, 'letzter_umsatz'] = df_langsmith.at[idx, 'letzter_umsatz'].replace("'", '"')

        lieferung = graph.query(query_finde_transport)
        df_langsmith.at[idx, 'transport'] = str(lieferung)
        df_langsmith.at[idx, 'transport'] = df_langsmith.at[idx, 'transport'].replace("'", '"')

        # reihengeschäft_prüfung = graph.query(query_reihengeschäft_prüfen)
        # df_langsmith.at[idx, 'reihengeschäft_prüfung'] = str(reihengeschäft_prüfung)
        # df_langsmith.at[idx, 'reihengeschäft_prüfung'] = df_langsmith.at[idx, 'reihengeschäft_prüfung'].replace("'", '"')

        transportverantwortung = graph.query(query_transportverantwortung)
        df_langsmith.at[idx, 'transportverantwortung'] = str(transportverantwortung)
        df_langsmith.at[idx, 'transportverantwortung'] = df_langsmith.at[idx, 'transportverantwortung'].replace("'", '"')

    except Exception as e:
        print(f"An unexpected error occurred: {e}")

df_langsmith.insert(loc=15, column='prüfung',value='')
timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M')
df_langsmith.to_excel(f'dataframes/{timestamp}_{filename}_df_with_query_outputs.xlsx', index=True)


#
# # NetworkX-Graph erstellen
# G = nx.DiGraph()  # Gerichteter Graph (falls ungerichtet: nx.Graph())
#
# # Knoten & Kanten hinzufügen
# for record in data:
#     G.add_edge(record["source"], record["target"], label=record["relationship"])
#
# # Zeichnen mit Labels
# plt.figure(figsize=(10, 7))
# pos = nx.spring_layout(G)  # Automatische Anordnung der Knoten
# nx.draw(G, pos, with_labels=True, node_color="lightblue", edge_color="gray", node_size=2000, font_size=10)
#
# # Beziehungstypen als Labels auf den Kanten anzeigen
# edge_labels = {(u, v): d["label"] for u, v, d in G.edges(data=True)}
# nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)
#
# plt.savefig("graph.png", dpi=300)
#
# print(edge_labels)
# # Knoten ausgeben
# print("Knoten:", list(G.nodes))
#
# # Kanten ausgeben
# print("Kanten:", list(G.edges(data=True)))

main()

if __name__ == "__main__":
    main()
