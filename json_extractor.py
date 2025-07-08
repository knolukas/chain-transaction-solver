import json
import os
import ast
import re
import sys

import pandas as pd
from dotenv import load_dotenv
from langchain_neo4j import Neo4jGraph
from langsmith import Client
from graphviz import Digraph
from selenium.webdriver.support.expected_conditions import element_selection_state_to_be

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


def replace_name_fields(text):
    counter = 1

    def replacer(match):
        nonlocal counter
        replacement = f"Name:'{counter}'"
        counter += 1
        return replacement

    # Regex sucht nach Name:'', Name: "", Name: ''
    new_text = re.sub(r"Name:\s*['\"]{2}", replacer, text)
    return new_text

def visualize_graph(graph, tv_name, id_internal, example_name):
    print("enter visualize_graph")
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

    # print(nodes)

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

    # print(edges)

    # Graph definieren
    dot = Digraph(format='pdf')
    dot.attr(rankdir='LR', size='10')  # Horizontal (Left-to-Right)

    # Knoten hinzufügen
    for node_id, props in nodes.items():
        # print('tv_name=' + tv_name)
        # print('node_id=' + node_id)
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
    dot.render(f"{id_internal}_{example_name}_visual_repr", format='pdf', cleanup=True)

    print(f"visualisation done, id {id_internal}_{example_name}")


query_letztes_unternehmen = """
OPTIONAL MATCH (n:Unternehmen)-[:BESTELLUNG]->() 
WHERE NOT n:Transportverantwortung AND NOT EXISTS { MATCH ()-[:BESTELLUNG]->(n) } 

RETURN COALESCE(n, "Inconsistent, no solution.") AS result
"""

query_erstes_unternehmen = """
OPTIONAL MATCH ()-[:BESTELLUNG]->(n:Unternehmen)
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

#only-jelly-47_REALWORLD.csv
#memorable-muscle-25_DUPONT.csv
filename = "pg__voo5__gpt-4.1__5bc806a6.csv"
df_langsmith = pd.read_csv("H:/Users/Lukas/OneDrive/Masterarbeit - LLMs in VAT - Knogler Lukas/Ergebnisse/" + filename)

reference_filename = "Beispiele_RG_test_set.xlsx"
df_database = pd.read_excel("H:/Users/Lukas/OneDrive/Masterarbeit - LLMs in VAT - Knogler Lukas/" + reference_filename)

df_merged = df_langsmith.merge(df_database[['id','internal_id', 'musterlösung_bewegte_lieferung', 'quelle', 'name']], how="left",on='id')

df_langsmith = df_merged.copy()

graph = Neo4jGraph(refresh_schema=False)

full_df = df_langsmith.index
#selection = range(13, 14)


for idx in full_df:
    try:
        #idx = 9
        delete_graph()
        df_langsmith.at[idx, 'outputs_decoded'] = df_langsmith.at[idx, 'outputs'].encode().decode('unicode_escape')
        df_langsmith.at[idx, 'inputs_decoded'] = df_langsmith.at[idx, 'inputs'].encode().decode('unicode_escape')
        print(f"\nstart with id: {idx}")
        try:
            data = json.loads(df_langsmith.at[idx, 'outputs'])
            #print(data)
        except json.JSONDecodeError as e:
            print(f"JSONDecodeError: {e}")
            continue

        quelle = df_langsmith.at[idx, 'quelle']
        name = df_langsmith.at[idx, 'name']
        example_name = f"{quelle}_{name}"
        id_internal = df_langsmith.at[idx, 'internal_id']

        cypher_statement = data['Cypher Anweisungen']
        cypher_statement = cypher_statement.replace("\\n", "\n")
        cypher_statement = cypher_statement.replace(";", "")
        cypher_statement = cypher_statement.replace("name", "Name")
        cypher_statement = cypher_statement.replace("sitz", "Sitz")
        cypher_statement = cypher_statement.replace("uid", "UID")
        cypher_statement = re.sub(r":(?:[A-Za-z0-9_]+)?(Transportverantwortung)", r":\1", cypher_statement)
        cypher_statement = replace_name_fields(cypher_statement)
        #print(cypher_statement)
        df_langsmith.at[idx, 'cypher_statements'] = cypher_statement
        graph_data = build_graph(cypher_statement)

        erstes_unternehmen = graph.query(query_erstes_unternehmen)
        df_langsmith.at[idx, 'erstes Unternehmen'] = str(erstes_unternehmen)
        df_langsmith.at[idx, 'erstes Unternehmen'] = df_langsmith.at[idx, 'erstes Unternehmen'].replace("'", '"')
        erstes_unternehmen_data = json.loads(df_langsmith.at[idx, 'erstes Unternehmen'])
        erstes_unternehmen_name = erstes_unternehmen_data[0]['result']['Name']

        letztes_unternehmen = graph.query(query_letztes_unternehmen)
        df_langsmith.at[idx, 'letztes Unternehmen'] = str(letztes_unternehmen)
        df_langsmith.at[idx, 'letztes Unternehmen'] = df_langsmith.at[idx, 'letztes Unternehmen'].replace("'", '"')
        letztes_unternehmen_data = json.loads(df_langsmith.at[idx, 'letztes Unternehmen'])
        letztes_unternehmen_name = letztes_unternehmen_data[0]['result']['Name']

        query_erster_umsatz = f"""
           //variable für Name "erstes Unternehmen"
           OPTIONAL MATCH (n:Unternehmen)-[:BESTELLUNG]->(m:Unternehmen {{Name: "{erstes_unternehmen_name}"}})
           RETURN n, 'BESTELLUNG' as Info, m"""
        erster_umsatz = graph.query(query_erster_umsatz)
        df_langsmith.at[idx, 'erster_umsatz'] = str(erster_umsatz)
        df_langsmith.at[idx, 'erster_umsatz'] = df_langsmith.at[idx, 'erster_umsatz'].replace("'", '"')

        query_letzter_umsatz = f"""
            //variable für Name "letztes Unternehmen"
            OPTIONAL MATCH (n:Unternehmen {{Name: "{letztes_unternehmen_name}"}})-[:BESTELLUNG]->(m:Unternehmen)
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
        anzahl_transportverantwortungen = graph.query("MATCH (:Unternehmen)-[r:HAT]-(:Transportverantwortung) RETURN count(r) AS anzahl_transportverantwortungen")
        df_langsmith.at[idx, 'bewegte lieferung'] = "Nicht gefunden."


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
        if anzahl_transportverantwortungen[0]['anzahl_transportverantwortungen'] != 1:
            error_flag = True
            error_text = error_text + "Transportverantwortung nicht eindeutig.\n"
            tv_name = "nicht definiert"
        else:
            tv_name = transportverantwortung[0]['n']['Name']
        if error_flag:
            df_langsmith.at[idx, 'ist_reihengeschäft'] = "False"
            df_langsmith.at[idx, 'ist_reihengeschäft_anmerkung'] = error_text
            bewegte_lieferung_query = f'\n//{error_text}'
            df_langsmith.at[idx, 'cypher_statements'] = (df_langsmith.at[idx, 'cypher_statements']
                                                         + ';\n'
                                                         + bewegte_lieferung_query)
            df_langsmith.at[idx, 'ergebnis_bewegte_lieferung'] = f"Kein RG erkannt"

        else:
            df_langsmith.at[idx, 'ist_reihengeschäft'] = "True"
            df_langsmith.at[idx, 'ist_reihengeschäft_anmerkung'] = error_text

            bewegte_lieferung_query = "//Bewegte Lieferung nicht definiert"
            start = "Start"
            ziel = "Ziel"

            if tv_name == letztes_unternehmen_name:
                df_langsmith.at[idx, 'bewegte lieferung'] = "Letzter Umsatz \n" + str(letzter_umsatz)
                start = letzter_umsatz[0]['m']['Name']
                ziel = letzter_umsatz[0]['n']['Name']

            elif tv_name == erstes_unternehmen_name:
                df_langsmith.at[idx, 'bewegte lieferung'] = "Erster Umsatz \n" + str(erster_umsatz)
                start = erster_umsatz[0]['m']['Name']
                ziel = erster_umsatz[0]['n']['Name']

            else:
                query_zwischen_umsatz = f"""
                            //variable für Name "zwischenhändler"
                            OPTIONAL MATCH (n:Unternehmen)-[:BESTELLUNG]->(m:Unternehmen {{Name: "{tv_name}"}})
                            RETURN n, 'BESTELLUNG' as Info, m"""
                zwischen_umsatz = graph.query(query_zwischen_umsatz)
                # df_langsmith.at[idx, 'zwischen_umsatz'] = str(zwischen_umsatz)
                # df_langsmith.at[idx, 'zwischen_umsatz'] = df_langsmith.at[idx, 'zwischen_umsatz'].replace("'", '"')


                query_vorangegangener_umsatz = f"""
                                            //variable für Name "zwischenhändler"
                                            OPTIONAL MATCH (n:Unternehmen {{Name: "{tv_name}"}})-[:BESTELLUNG]->(m:Unternehmen)
                                            RETURN n, 'BESTELLUNG' as Info, m"""
                vorangegangener_umsatz = graph.query(query_vorangegangener_umsatz)

                abgangsstaat = graph.query(query_finde_abgangsstaat)
                abgangsstaat = abgangsstaat[0]['n.Sitz']
                tv_uid = transportverantwortung[0]['n']['UID'][:2]

                if tv_uid == abgangsstaat:
                    df_langsmith.at[idx, 'bewegte lieferung'] = (("Ja, UID gleich wie Abgangsstaat."
                                                                  f"\n{tv_uid} == {abgangsstaat}"
                                                                 "\nUmsatz des Zwischenhändlers\n")
                                                                 + str(zwischen_umsatz))
                    start = zwischen_umsatz[0]['m']['Name']
                    ziel = zwischen_umsatz[0]['n']['Name']
                else:
                    df_langsmith.at[idx, 'bewegte lieferung'] = (("Andere oder gar keine UID."
                                                                  f"\n{tv_uid} != {abgangsstaat}"
                                                                 "\nUmsatz zum Zwischenhändler. \n")
                                                                 + str(vorangegangener_umsatz))
                    start = vorangegangener_umsatz[0]['m']['Name']
                    ziel = vorangegangener_umsatz[0]['n']['Name']


            bewegte_lieferung_query = (f'\nMATCH (a:Unternehmen {{Name: "{start}"}}),'
                                       f'(b:Unternehmen {{Name: "{ziel}"}})'
                                       '\nCREATE (a)-[:BEWEGTE_LIEFERUNG]->(b)')
            df_langsmith.at[idx, 'cypher_statements'] = (df_langsmith.at[idx, 'cypher_statements']
                                                         + ';\n'
                                                         + bewegte_lieferung_query)
            graph.query(bewegte_lieferung_query)

            df_langsmith.at[idx, 'ergebnis_bewegte_lieferung'] = f"{start}->{ziel}"


        if df_langsmith.at[idx, 'ergebnis_bewegte_lieferung'] == df_langsmith.at[idx, 'musterlösung_bewegte_lieferung']:
            df_langsmith.at[idx, 'vergleich'] = "richtig"
        else:
            df_langsmith.at[idx, 'vergleich'] = "prüfung"

        visualize_graph(graph, tv_name, id_internal, example_name)

    except Exception as e:
        print(f"An unexpected error occurred: {e}, at ID:{idx}")
        e_type, e_object, e_traceback = sys.exc_info()

        e_filename = os.path.split(
            e_traceback.tb_frame.f_code.co_filename
        )[1]

        e_message = str(e)

        e_line_number = e_traceback.tb_lineno

        print(f'exception type: {e_type}')

        print(f'exception filename: {e_filename}')

        print(f'exception line number: {e_line_number}')

        print(f'exception message: {e_message}')

timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M')
df_langsmith.to_excel(f'H:/Users/Lukas/OneDrive/Masterarbeit - LLMs in VAT - Knogler Lukas/Ergebnisse/{filename}_{timestamp}_df_with_query_outputs.xlsx', index=True)

