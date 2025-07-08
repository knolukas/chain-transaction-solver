"""
This module is used to generate the KG from provided data.
It also applies logic-based rules to identify the movable supply.
It stores the data and a visualization.
"""

import json
import os
import re
import sys
import pandas as pd
from dotenv import load_dotenv
from langchain_neo4j import Neo4jGraph
from graphviz import Digraph

#Initialize credentials and access
load_dotenv('.env')
os.environ["NEO4J_URI"] = os.getenv("NEO4J_URI")
os.environ["NEO4J_USERNAME"] = os.getenv("NEO4J_USERNAME")
os.environ["NEO4J_PASSWORD"] = os.getenv("NEO4J_PASSWORD")
LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY")

#Global variable
tv_name = "nicht definiert"

#Define functions
def delete_graph():
    """
    Delete the graph from Neo4j
    :return: none
    """
    query = "MATCH (n) DETACH DELETE n"
    graph.query(query)
    print("Graph deleted successfully")

def build_graph(query):
    """
    Build the graph in Neo4j
    :param query: Statements to build the graph
    :return: Graph data
    """
    data = graph.query(query)
    print("Graph built successfully")
    return data

def replace_name_fields(text):
    """
    Replace name field to remove inconsistencies
    :param text: Text to clean
    :return: cleaned text
    """
    counter = 1

    def replacer(match):
        nonlocal counter
        replacement = f"Name:'{counter}'"
        counter += 1
        return replacement

    # Regex sucht nach Name:'', Name: "", Name: ''
    new_text = re.sub(r"Name:\s*['\"]{2}", replacer, text)
    return new_text

def clean_statement(statement):
    """
    Clean statement to remove inconsistencies
    :param statement: Statement to clean
    :return: Cleaned statement
    """
    statement = (statement.replace("\\n", "\n")
                 .replace(";", "")
                 .replace("name", "Name")
                 .replace("sitz", "Sitz")
                 .replace("uid", "UID"))
    statement = re.sub(r":(?:[A-Za-z0-9_]+)?(Transportverantwortung)", r":\1", statement)
    statement = replace_name_fields(statement)
    print(statement)
    return statement

def apply_logic_based_rules(graph, df):
    """
    Application of the logic based rules.

    First, run queries to identify and extract required data.
    Second, check whether there is a valid chain transaction.
    Third, identify the movable supply.

    :param graph: Knowledge graph from the KG-database
    :param df: Dataframe to store the output
    :return: none
    """

    global tv_name

    ###########################################################
    #First, run queries to identify and extract required data #
    ###########################################################

    erstes_unternehmen = graph.query(query_erstes_unternehmen)
    df.at[idx, 'erstes Unternehmen'] = str(erstes_unternehmen)
    df.at[idx, 'erstes Unternehmen'] = df.at[idx, 'erstes Unternehmen'].replace("'", '"')
    erstes_unternehmen_data = json.loads(df.at[idx, 'erstes Unternehmen'])
    erstes_unternehmen_name = erstes_unternehmen_data[0]['result']['Name']

    letztes_unternehmen = graph.query(query_letztes_unternehmen)
    df.at[idx, 'letztes Unternehmen'] = str(letztes_unternehmen)
    df.at[idx, 'letztes Unternehmen'] = df.at[idx, 'letztes Unternehmen'].replace("'", '"')
    letztes_unternehmen_data = json.loads(df.at[idx, 'letztes Unternehmen'])
    letztes_unternehmen_name = letztes_unternehmen_data[0]['result']['Name']

    query_erster_umsatz = f"""
               //variable für Name "erstes Unternehmen"
               OPTIONAL MATCH (n:Unternehmen)-[:BESTELLUNG]->(m:Unternehmen {{Name: "{erstes_unternehmen_name}"}})
               RETURN n, 'BESTELLUNG' as Info, m"""
    erster_umsatz = graph.query(query_erster_umsatz)
    df.at[idx, 'erster_umsatz'] = str(erster_umsatz)
    df.at[idx, 'erster_umsatz'] = df.at[idx, 'erster_umsatz'].replace("'", '"')

    query_letzter_umsatz = f"""
                //variable für Name "letztes Unternehmen"
                OPTIONAL MATCH (n:Unternehmen {{Name: "{letztes_unternehmen_name}"}})-[:BESTELLUNG]->(m:Unternehmen)
                RETURN n, 'BESTELLUNG' as Info, m"""
    letzter_umsatz = graph.query(query_letzter_umsatz)
    df.at[idx, 'letzter_umsatz'] = str(letzter_umsatz)
    df.at[idx, 'letzter_umsatz'] = df.at[idx, 'letzter_umsatz'].replace("'", '"')

    warenbewegung = graph.query(query_finde_warenbewegung)
    df.at[idx, 'warenbewegung'] = str(warenbewegung)
    df.at[idx, 'warenbewegung'] = df.at[idx, 'warenbewegung'].replace("'", '"')
    start_warenbewegung_name = warenbewegung[0]['n']['Name']
    ziel_warenbewegung_name = warenbewegung[0]['m']['Name']

    transportverantwortung = graph.query(query_transportverantwortung)
    df.at[idx, 'transportverantwortung'] = str(transportverantwortung)
    df.at[idx, 'transportverantwortung'] = df.at[idx, 'transportverantwortung'].replace("'", '"')

    anzahl_bestellungen = graph.query(
        "MATCH (:Unternehmen)-[r:BESTELLUNG]->(:Unternehmen) RETURN count(r) AS anzahl_bestellungen")
    anzahl_unternehmen = graph.query("MATCH (n:Unternehmen) RETURN count(n) AS anzahl_unternehmen")
    anzahl_warenbewegungen = graph.query(
        "MATCH (:Unternehmen)-[r:WARENBEWEGUNG]->(:Unternehmen) RETURN count(r) AS anzahl_warenbewegungen")
    anzahl_unterschiedlicher_produkte = graph.query(query_anzahl_produkte)
    anzahl_transportverantwortungen = graph.query(
        "MATCH (:Unternehmen)-[r:HAT]-(:Transportverantwortung) RETURN count(r) AS anzahl_transportverantwortungen")
    df.at[idx, 'bewegte lieferung'] = "Nicht gefunden."


    ###########################################################
    #Second, check whether there is a valid chain transaction #
    ###########################################################
    error_flag = False
    error_text = ""
    if not (
            erstes_unternehmen_name == start_warenbewegung_name and letztes_unternehmen_name == ziel_warenbewegung_name):
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
        df.at[idx, 'ist_reihengeschäft'] = "False"
        df.at[idx, 'ist_reihengeschäft_anmerkung'] = error_text
        bewegte_lieferung_query = f'\n//{error_text}'
        df.at[idx, 'cypher_statements'] = (df.at[idx, 'cypher_statements']
                                                     + ';\n'
                                                     + bewegte_lieferung_query)
        df.at[idx, 'ergebnis_bewegte_lieferung'] = f"Kein RG erkannt"


    #####################################
    #Third, identify the movable supply #
    #####################################
    else:
        df.at[idx, 'ist_reihengeschäft'] = "True"
        df.at[idx, 'ist_reihengeschäft_anmerkung'] = error_text

        if tv_name == letztes_unternehmen_name:
            df.at[idx, 'bewegte lieferung'] = "Letzter Umsatz \n" + str(letzter_umsatz)
            start = letzter_umsatz[0]['m']['Name']
            ziel = letzter_umsatz[0]['n']['Name']

        elif tv_name == erstes_unternehmen_name:
            df.at[idx, 'bewegte lieferung'] = "Erster Umsatz \n" + str(erster_umsatz)
            start = erster_umsatz[0]['m']['Name']
            ziel = erster_umsatz[0]['n']['Name']

        else:
            query_zwischen_umsatz = f"""
                                //variable für Name "zwischenhändler"
                                OPTIONAL MATCH (n:Unternehmen)-[:BESTELLUNG]->(m:Unternehmen {{Name: "{tv_name}"}})
                                RETURN n, 'BESTELLUNG' as Info, m"""
            zwischen_umsatz = graph.query(query_zwischen_umsatz)
            # df.at[idx, 'zwischen_umsatz'] = str(zwischen_umsatz)
            # df.at[idx, 'zwischen_umsatz'] = df.at[idx, 'zwischen_umsatz'].replace("'", '"')

            query_vorangegangener_umsatz = f"""
                                                //variable für Name "zwischenhändler"
                                                OPTIONAL MATCH (n:Unternehmen {{Name: "{tv_name}"}})-[:BESTELLUNG]->(m:Unternehmen)
                                                RETURN n, 'BESTELLUNG' as Info, m"""
            vorangegangener_umsatz = graph.query(query_vorangegangener_umsatz)

            abgangsstaat = graph.query(query_finde_abgangsstaat)
            abgangsstaat = abgangsstaat[0]['n.Sitz']
            tv_uid = transportverantwortung[0]['n']['UID'][:2]

            if tv_uid == abgangsstaat:
                df.at[idx, 'bewegte lieferung'] = (("Ja, UID gleich wie Abgangsstaat."
                                                              f"\n{tv_uid} == {abgangsstaat}"
                                                              "\nUmsatz des Zwischenhändlers\n")
                                                             + str(zwischen_umsatz))
                start = zwischen_umsatz[0]['m']['Name']
                ziel = zwischen_umsatz[0]['n']['Name']
            else:
                df.at[idx, 'bewegte lieferung'] = (("Andere oder gar keine UID."
                                                              f"\n{tv_uid} != {abgangsstaat}"
                                                              "\nUmsatz zum Zwischenhändler. \n")
                                                             + str(vorangegangener_umsatz))
                start = vorangegangener_umsatz[0]['m']['Name']
                ziel = vorangegangener_umsatz[0]['n']['Name']

        bewegte_lieferung_query = (f'\nMATCH (a:Unternehmen {{Name: "{start}"}}),'
                                   f'(b:Unternehmen {{Name: "{ziel}"}})'
                                   '\nCREATE (a)-[:BEWEGTE_LIEFERUNG]->(b)')
        df.at[idx, 'cypher_statements'] = (df.at[idx, 'cypher_statements']
                                                     + ';\n'
                                                     + bewegte_lieferung_query)
        ################################################
        #Add the movable supply to the knowledge graph #
        ################################################
        graph.query(bewegte_lieferung_query)
        df.at[idx, 'ergebnis_bewegte_lieferung'] = f"{start}->{ziel}"

def visualize_graph(graph, tv_name, id_internal, example_name):
    """
    Custom visualization of the graph
    :param graph: Knowledge graph
    :param tv_name: Name of transport responsible enterprise
    :param id_internal: Internal ID of the case
    :param example_name: Name of the case
    :return: none
    """
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


    # Graph definieren
    dot = Digraph(format='pdf')
    dot.attr(rankdir='LR', size='10')  # Horizontal (Left-to-Right)

    # Knoten hinzufügen
    for node_id, props in nodes.items():

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

########################################
#Queries to extract required knowledge #
########################################
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


#Load data

#only-jelly-47_REALWORLD.csv
#memorable-muscle-25_DUPONT.csv
#glossy-question-8_MuW.csv
#pg__voo5__gpt-4.1__39c2a515_KOLLMANN.csv

filename = "pg__voo5__gpt-4.1__5bc806a6.csv"
df_langsmith = pd.read_csv("H:/Users/Lukas/OneDrive/Masterarbeit - LLMs in VAT - Knogler Lukas/Ergebnisse/" + filename)
reference_filename = "Beispiele_RG_test_set.xlsx"
df_database = pd.read_excel("H:/Users/Lukas/OneDrive/Masterarbeit - LLMs in VAT - Knogler Lukas/" + reference_filename)
df_merged = df_langsmith.merge(df_database[['id','internal_id', 'musterlösung_bewegte_lieferung', 'quelle', 'name']], how="left",on='id')
df_langsmith = df_merged.copy()

#Initialize KG DB
graph = Neo4jGraph(refresh_schema=False)

full_df = df_langsmith.index
#selection = range(13, 14)


for idx in full_df:
    try:
        #idx = 9

        #Initialize and clean the database#
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

        #Construct distinct names for cases#
        quelle = df_langsmith.at[idx, 'quelle']
        name = df_langsmith.at[idx, 'name']
        example_name = f"{quelle}_{name}"
        id_internal = df_langsmith.at[idx, 'internal_id']

        cypher_statement = data['Cypher Anweisungen']
        cypher_statement = clean_statement(cypher_statement)

        df_langsmith.at[idx, 'cypher_statements'] = cypher_statement
        graph_data = build_graph(cypher_statement)

        #Apply logic based rules#
        apply_logic_based_rules(graph, df_langsmith)


        #Check if result equals the sample solution#
        #Add 'Prüfung' if check is not successful  #
        if df_langsmith.at[idx, 'ergebnis_bewegte_lieferung'] == df_langsmith.at[idx, 'musterlösung_bewegte_lieferung']:
            df_langsmith.at[idx, 'vergleich'] = "richtig"
        else:
            df_langsmith.at[idx, 'vergleich'] = "prüfung"

        #Generate visualization#
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

############################
#Store the result as .xlsx #
############################
timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M')
df_langsmith.to_excel(f'H:/Users/Lukas/OneDrive/Masterarbeit - LLMs in VAT - Knogler Lukas/Ergebnisse/{filename}_{timestamp}_df_with_query_outputs.xlsx', index=True)

