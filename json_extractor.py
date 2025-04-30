import json
import os
import ast

import pandas as pd
from dotenv import load_dotenv
from langchain_neo4j import Neo4jGraph
from langsmith import Client
from graphviz import Digraph
from tax_scraper import get_tax_rate

#tax_rate = get_tax_rate("austria")
#def main():
load_dotenv('.env')
os.environ["NEO4J_URI"] = os.getenv("NEO4J_URI")
os.environ["NEO4J_USERNAME"] = os.getenv("NEO4J_USERNAME")
os.environ["NEO4J_PASSWORD"] = os.getenv("NEO4J_PASSWORD")
LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY")

def delete_graph():
    query = "MATCH (n) DETACH DELETE n"
    graph.query(query)
    print("Graph deleted successfully")

def build_graph(query):
    data = graph.query(query)
    print("Graph built successfully")
    return data

query_endkunde = """
OPTIONAL MATCH (n) 
WHERE NOT n:Transportverantwortung AND NOT EXISTS { MATCH ()-[:BESTELLUNG]->(n) } 

RETURN COALESCE(n, "Inconsistent, no solution.") AS result
"""

query_hersteller = """
OPTIONAL MATCH (n) 
WHERE NOT n:Transportverantwortung AND NOT EXISTS { MATCH (n)-[:BESTELLUNG]->() } 

RETURN COALESCE(n, "Inconsistent, no solution.") AS result"""

query_finde_warenbewegung = """
    //Finde Lieferung
    OPTIONAL MATCH (n:Unternehmen)-[:WARENBEWEGUNG]->(m:Unternehmen) 
    RETURN n, 'WARENBEWEGUNG' as Info, m
    """

query_finde_abgangsstaat = """
    //Finde Lieferung
    OPTIONAL MATCH (n:Unternehmen)-[:WARENBEWEGUNG]->(m:Unternehmen) 
    RETURN n.Sitz
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

query_anzahl_produkte = """
    MATCH ()-[r]->()
    WHERE type(r) IN ['BESTELLUNG', 'WARENBEWEGUNG']
    RETURN count(DISTINCT r.Produkt) AS anzahl_unterschiedlicher_produkte"""


filename = "pg__voo5__gpt-4.1__c0c14ef7.csv"
df_langsmith = pd.read_csv("H:/Users/Lukas/Downloads/" + filename)
# internal_ids = [14, 9, 31, 13, 7, 6, 4, 1, 10, 35, 17, 30, 27, 5, 18, 32, 34, 2, 36, 20, 29, 28, 37, 38, 3, 39, 19, 11,
#                 8, 15, 25, 33, 26, 24, 23, 12, 16, 22, 21]
#
# quelle = [
#     "KOLLMANN", "KOLLMANN", "KOLLMANN", "KOLLMANN", "KOLLMANN",
#     "KOLLMANN", "KOLLMANN", "KOLLMANN", "KOLLMANN", "KOLLMANN",
#     "KOLLMANN", "KOLLMANN", "KOLLMANN", "KOLLMANN", "KOLLMANN",
#     "KOLLMANN", "KOLLMANN", "KOLLMANN", "KOLLMANN", "KOLLMANN",
#     "REALCASES", "REALCASES", "REALCASES", "REALCASES", "REALCASES",
#     "REALCASES", "REALCASES", "REALCASES", "REALCASES", "REALCASES",
#     "REALCASES", "REALCASES", "REALCASES", "REALCASES", "REALCASES",
#     "REALCASES", "REALCASES", "REALCASES", "REALCASES"
# ]
# df_langsmith.insert(loc=0,column='interne_ID',value=internal_ids)
# df_langsmith = df_langsmith.sort_values(by=['interne_ID'], ascending=[True])
# df_langsmith.insert(loc=1,column='quelle',value=quelle)
# df_langsmith = df_langsmith.sort_index(ascending=[True])
graph = Neo4jGraph(refresh_schema=False)



full_df = df_langsmith.index
selection = range(12, 13)


for idx in full_df:
    try:
        #idx = 0
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

        erstes_unternehmen = graph.query(query_hersteller)
        df_langsmith.at[idx, 'erstes Unternehmen'] = str(erstes_unternehmen)
        df_langsmith.at[idx, 'erstes Unternehmen'] = df_langsmith.at[idx, 'erstes Unternehmen'].replace("'", '"')
        erstes_unternehmen_data = json.loads(df_langsmith.at[idx, 'erstes Unternehmen'])
        erstes_unternehmen_name = erstes_unternehmen_data[0]['result']['Name']

        letztes_unternehmen = graph.query(query_endkunde)
        df_langsmith.at[idx, 'letztes Unternehmen'] = str(letztes_unternehmen)
        df_langsmith.at[idx, 'letztes Unternehmen'] = df_langsmith.at[idx, 'letztes Unternehmen'].replace("'", '"')
        letztes_unternehmen_data = json.loads(df_langsmith.at[idx, 'letztes Unternehmen'])
        letztes_unternehmen_name = letztes_unternehmen_data[0]['result']['Name']

        query_erster_umsatz = f"""
           //variable für Name "erstes Unternehmen"
           OPTIONAL MATCH (n:Unternehmen {{Name: "{erstes_unternehmen_name}"}})-[:BESTELLUNG]-(m:Unternehmen)
           RETURN n, 'BESTELLUNG' as Info, m"""
        erster_umsatz = graph.query(query_erster_umsatz)
        df_langsmith.at[idx, 'erster_umsatz'] = str(erster_umsatz)
        df_langsmith.at[idx, 'erster_umsatz'] = df_langsmith.at[idx, 'erster_umsatz'].replace("'", '"')

        query_letzter_umsatz = f"""
            //variable für Name "letztes Unternehmen"
            OPTIONAL MATCH (n:Unternehmen)-[:BESTELLUNG]-(m:Unternehmen {{Name: "{letztes_unternehmen_name}"}})
            RETURN n, 'BESTELLUNG' as Info, m"""
        letzter_umsatz = graph.query(query_letzter_umsatz)
        df_langsmith.at[idx, 'letzter_umsatz'] = str(letzter_umsatz)
        df_langsmith.at[idx, 'letzter_umsatz'] = df_langsmith.at[idx, 'letzter_umsatz'].replace("'", '"')

        warenbewegung = graph.query(query_finde_warenbewegung)
        df_langsmith.at[idx, 'warenbewegung'] = str(warenbewegung)
        df_langsmith.at[idx, 'warenbewegung'] = df_langsmith.at[idx, 'warenbewegung'].replace("'", '"')
        start_warenbewegung_name = warenbewegung[0]['n']['Name']
        ziel_warenbewegung_name = warenbewegung[0]['m']['Name']

        transportverantwortung = graph.query(query_transportverantwortung)
        df_langsmith.at[idx, 'transportverantwortung'] = str(transportverantwortung)
        df_langsmith.at[idx, 'transportverantwortung'] = df_langsmith.at[idx, 'transportverantwortung'].replace("'", '"')

        anzahl_bestellungen = graph.query("MATCH (:Unternehmen)-[r:BESTELLUNG]->(:Unternehmen) RETURN count(r) AS anzahl_bestellungen")
        anzahl_unternehmen = graph.query("MATCH (n:Unternehmen) RETURN count(n) AS anzahl_unternehmen")
        anzahl_warenbewegungen = graph.query("MATCH (:Unternehmen)-[r:WARENBEWEGUNG]->(:Unternehmen) RETURN count(r) AS anzahl_warenbewegungen")
        anzahl_unterschiedlicher_produkte = graph.query(query_anzahl_produkte)

        error_flag = False
        error_text = ""
        if not(erstes_unternehmen_name == start_warenbewegung_name and letztes_unternehmen_name == ziel_warenbewegung_name):
            error_flag = True
            error_text = error_text + "Keine Übergabe vom letzten zum ersten Unternehmen.\n"
        if anzahl_bestellungen[0]['anzahl_bestellungen'] < 2:
            error_flag = True
            error_text = error_text + "Weniger als 2 Bestellungen.\n"
        if anzahl_unternehmen[0]['anzahl_unternehmen'] < 3:
            error_flag = True
            error_text = error_text + "Es sind weniger als 3 Unternehmen involviert.\n"
        if anzahl_unterschiedlicher_produkte[0]['anzahl_unterschiedlicher_produkte'] > 1:
            error_flag = True
            error_text = error_text + "Es werden unterschiedliche Produkte gehandelt.\n"
        if anzahl_warenbewegungen[0]['anzahl_warenbewegungen'] != 1:
            error_flag = True
            error_text = error_text + "Es gibt mehrere Warenbewegungen (oder keine).\n"
        if error_flag:
            df_langsmith.at[idx, 'ist_reihengeschäft'] = "False"
            df_langsmith.at[idx, 'ist_reihengeschäft_anmerkung'] = error_text
        else:
            df_langsmith.at[idx, 'ist_reihengeschäft'] = "True"
            df_langsmith.at[idx, 'ist_reihengeschäft_anmerkung'] = error_text

        transportverantwortung_str = df_langsmith.at[idx, 'transportverantwortung']
        bewegte_lieferung_query = "//Bewegte Lieferung nicht definiert"
        start = "Start"
        ziel = "Ziel"
        tv_name = "unknown"

        if transportverantwortung_str == "[]":
            df_langsmith.at[idx, 'bewegte lieferung'] = "Keine Transportverantwortung festgestellt. Prüfung notwendig."
            #continue
        else:
            tv_data = ast.literal_eval(transportverantwortung_str)
            if len(tv_data) > 1:
                df_langsmith.at[idx, 'bewegte lieferung'] = "Transportverantwortung nicht eindeutig. Prüfung notwendig"
                #continue
            tv_name = tv_data[0]['n']['Name']
            if tv_name == letztes_unternehmen_name:
                df_langsmith.at[idx, 'bewegte lieferung'] = "Letzter Umsatz \n" + str(letzter_umsatz)
                start = letzter_umsatz[0]['n']['Name']
                ziel = tv_name

            elif tv_name == erstes_unternehmen_name:
                df_langsmith.at[idx, 'bewegte lieferung'] = "Erster Umsatz \n" + str(erster_umsatz)
                start = tv_name
                ziel = letzter_umsatz[0]['n']['Name']

            else:
                query_zwischen_umsatz = f"""
                            //variable für Name "zwischenhändler"
                            OPTIONAL MATCH (n:Unternehmen)-[:BESTELLUNG]->(m:Unternehmen {{Name: "{tv_name}"}})
                            RETURN n, 'BESTELLUNG' as Info, m"""
                zwischen_umsatz = graph.query(query_zwischen_umsatz)
                df_langsmith.at[idx, 'zwischen_umsatz'] = str(zwischen_umsatz)
                df_langsmith.at[idx, 'zwischen_umsatz'] = df_langsmith.at[idx, 'zwischen_umsatz'].replace("'", '"')

                zwischen_data = ast.literal_eval(str(zwischen_umsatz))
                zwischen_name = zwischen_data[0]['m']['Name']



                query_vorangegangener_umsatz = f"""
                                            //variable für Name "zwischenhändler"
                                            OPTIONAL MATCH (n:Unternehmen {{Name: "{tv_name}"}})-[:BESTELLUNG]->(m:Unternehmen)
                                            RETURN n, 'BESTELLUNG' as Info, m"""
                vorangegangener_umsatz = graph.query(query_vorangegangener_umsatz)

                abgangsstaat = graph.query(query_finde_abgangsstaat)
                #print(query_finde_abgangsstaat)
                #print('\n')
                #print(abgangsstaat)

                tv_uid = tv_data[0]['n']['UID']


                # client = Client(api_key=LANGSMITH_API_KEY)
                # prompt = client.pull_prompt("uid_estimator", include_model=True)
                # response = prompt.invoke({"Land": f"{abgangsstaat}", "Nummer": f"{tv_uid}"})
                #print(response)

                if tv_uid == abgangsstaat:
                    df_langsmith.at[idx, 'bewegte lieferung'] = (("Ja, UID gleich wie Abgangsstaat."
                                                                  f"\n{tv_uid} == {abgangsstaat}"
                                                                 "\nUmsatz des Zwischenhändlers\n")
                                                                 + str(zwischen_umsatz))
                    start = zwischen_umsatz[0]['m']['Name']
                    ziel = zwischen_umsatz[0]['n']['Name']

                # elif tv_uid != abgangsstaat:
                #     df_langsmith.at[idx, 'bewegte lieferung'] = (("Nein, UID nicht gleich Abgangsstaat."
                #                                                  "\nUmsatz zum Zwischenhändler. \n")
                #                                                  + str(vorangegangener_umsatz))
                #     start = vorangegangener_umsatz[0]['m']['Name']
                #     ziel = vorangegangener_umsatz[0]['n']['Name']

                else:
                    df_langsmith.at[idx, 'bewegte lieferung'] = (("Unklar, andere oder gar keine UID."
                                                                  f"\n{tv_uid} != {abgangsstaat}"
                                                                 "\nUmsatz zum Zwischenhändler. \n")
                                                                 + str(vorangegangener_umsatz))
                    start = vorangegangener_umsatz[0]['m']['Name']
                    ziel = vorangegangener_umsatz[0]['n']['Name']


        bewegte_lieferung_query = (f'\nMATCH (a:Unternehmen {{Name: "{start}"}}),'
                                   f'(b:Unternehmen {{Name: "{ziel}"}})'
                                   '\nCREATE (a)-[:BEWEGTE_LIEFERUNG]->(b)')
        df_langsmith.at[idx, 'cypher_statements'] = (df_langsmith.at[idx, 'cypher_statements']
                                                     + ';'
                                                     + bewegte_lieferung_query)

        zwischenergebnis = graph.query(bewegte_lieferung_query)
        #print(zwischenergebnis)

        nodes = {}

        cypher_query_nodes = """
        MATCH (n)
        RETURN
          CASE
            WHEN n.Name IS NOT NULL THEN n.Name
            ELSE 'Unknown'
          END AS id,
          CASE
            WHEN n.Name IS NOT NULL THEN n.Name + '\\nSitz: ' + n.Sitz + '\\nUID: ' + n.UID
            WHEN n.Produkt IS NOT NULL THEN 'Transportverantwortung\\nProdukt: ' + n.Produkt
            ELSE 'Unbekannter Knoten'
          END AS label
        """

        result = graph.query(cypher_query_nodes)
        for record in result:
            if 'Transportverantwortung' in record["label"]:
                continue
            node_id = record["id"]
            label = record["label"]
            nodes[node_id] = {"label": label}

        #print(nodes)

        edges = []

        cypher_query_edges = """
        MATCH (a)-[r]->(b)
        RETURN
          CASE
            WHEN a.Name IS NOT NULL THEN a.Name
            WHEN a.Produkt IS NOT NULL THEN 'T'
            ELSE 'Unknown'
          END AS source,
          CASE
            WHEN b.Name IS NOT NULL THEN b.Name
            WHEN b.Produkt IS NOT NULL THEN 'T'
            ELSE 'Unknown'
          END AS target,
          TYPE(r) +
            CASE WHEN r.Produkt IS NOT NULL THEN '\\nProdukt: ' + r.Produkt ELSE '' END AS label
        """


        result = graph.query(cypher_query_edges)
        for record in result:
            if record["label"] == "HAT":
                continue
            source = record["source"]
            target = record["target"]
            label = record["label"]
            edges.append((source, target, label))

        #print(edges)

        # Graph definieren
        dot = Digraph(format='pdf')
        dot.attr(rankdir='LR', size='10')  # Horizontal (Left-to-Right)


        # Knoten hinzufügen
        for node_id, props in nodes.items():
            #print('tv_name=' + tv_name)
            #print('node_id=' + node_id)
            if node_id == tv_name:
                dot.node(node_id, label=props['label'], shape='box', color='red', style='filled', fillcolor='lightblue')
            else:
                dot.node(node_id, label=props['label'], shape='box', style='filled', fillcolor='lightblue')

        # Kanten hinzufügen
        for src, dst, label in edges:
            if 'BEWEGTE_LIEFERUNG' in label:
                dot.edge(src, dst, label=label, color='red', fontcolor='red')
            else:
                dot.edge(src, dst, label=label)

        # Als PDF speichern
        dot.render(f"neo4j_graph_id_{idx}", format='pdf', cleanup=True)



    except Exception as e:
        print(f"An unexpected error occurred: {e}, at ID:{idx}")

df_langsmith.insert(loc=15, column='prüfung',value='')
timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M')
df_langsmith.to_excel(f'dataframes/{timestamp}_{filename}_df_with_query_outputs.xlsx', index=True)