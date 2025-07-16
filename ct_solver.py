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
from graphviz import Digraph
from langchain_neo4j import Neo4jGraph

#Initialize credentials and access
load_dotenv('.env')
os.environ["NEO4J_URI"] = os.getenv("NEO4J_URI")
os.environ["NEO4J_USERNAME"] = os.getenv("NEO4J_USERNAME")
os.environ["NEO4J_PASSWORD"] = os.getenv("NEO4J_PASSWORD")
LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY")

#Global variable
tr_name = "not defined"

#Define functions
def delete_graph():
    """
    Delete the graph from Neo4j
    :return: none
    """
    query = "MATCH (n) DETACH DELETE n"
    graph.query(query)
    print("\n Graph deleted successfully")

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

    def replacer():
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
    #print(statement)
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

    global tr_name

    ###########################################################
    #First, run queries to identify and extract required data #
    ###########################################################

    first_enterprise = graph.query(query_first_enterprise)
    first_enterprise_name = first_enterprise[0]['result']['Name']
    df.at[idx, 'first_enterprise'] = str(first_enterprise_name)

    last_enterprise = graph.query(query_last_enterprise)
    last_enterprise_name = last_enterprise[0]['result']['Name']
    df.at[idx, 'last_enterprise'] = str(last_enterprise_name)

    transport_of_goods = graph.query(query_find_transport_of_goods)
    start_transport_name = transport_of_goods[0]['n']['Name']
    destination_transport_name = transport_of_goods[0]['m']['Name']
    df.at[idx, 'transport_of_goods'] = str(start_transport_name + "->" + destination_transport_name)

    no_of_orders = graph.query(query_no_of_orders)
    no_of_enterprises = graph.query(query_no_of_enterprises)
    no_of_transports_of_goods = graph.query(query_no_of_transports_of_goods)
    no_of_products = graph.query(query_no_of_products)
    no_of_tr = graph.query(query_no_of_tr)

    df.at[idx, 'moved_supply'] = "Not indetified."


    ###########################################################
    #Second, check whether there is a valid chain transaction #
    ###########################################################
    error_flag = False
    error_text = ""
    if not (
            first_enterprise_name == start_transport_name and last_enterprise_name == destination_transport_name):
        error_flag = True
        error_text = error_text + "No transport of goods between last and first enterprise.\n"
    if no_of_orders[0]['no_of_orders'] < 2:
        error_flag = True
        error_text = error_text + "Less than 2 orders.\n"
    if no_of_enterprises[0]['no_of_enterprises'] < 3:
        error_flag = True
        error_text = error_text + "Less than 3 enterprises.\n"
    if no_of_products[0]['no_of_products'] > 1:
        error_flag = True
        error_text = error_text + "Different products traded.\n"
    if no_of_transports_of_goods[0]['no_of_transports_of_goods'] != 1:
        error_flag = True
        error_text = error_text + "More than one transport of goods (or none).\n"
    if no_of_tr[0]['no_of_tr'] != 1:
        error_flag = True
        error_text = error_text + "Transport responsibility is not defined.\n"
        tr_name = "not defined"
    else:
        transport_responsibility = graph.query(query_transport_responsibility)
        tr_name = transport_responsibility[0]['n']['Name']

    df.at[idx, 'transport_responsibility'] = str(tr_name)

    if error_flag:
        df.at[idx, 'is_chain_transaction'] = "False"
        df.at[idx, 'is_chain_transaction_comment'] = error_text
        query_movable_supply = f'\n//{error_text}'
        df.at[idx, 'cypher_statements'] = (df.at[idx, 'cypher_statements']
                                                     + ';\n'
                                                     + query_movable_supply)
        df.at[idx, 'identified_movable_supply'] = f"No CT identified"


    #####################################
    #Third, identify the movable supply #
    #####################################

    else:
        query_first_supply = f"""
                       OPTIONAL MATCH (n:Unternehmen)-[:BESTELLUNG]->(m:Unternehmen {{Name: "{first_enterprise_name}"}})
                       RETURN n, 'BESTELLUNG' as Info, m"""
        first_supply = graph.query(query_first_supply)
        df.at[idx, 'first_supply'] = str(first_supply[0]['m']['Name'] + "->" + first_supply[0]['n']['Name'])

        query_last_supply = f"""
                        OPTIONAL MATCH (n:Unternehmen {{Name: "{last_enterprise_name}"}})-[:BESTELLUNG]->(m:Unternehmen)
                        RETURN n, 'BESTELLUNG' as Info, m"""
        last_supply = graph.query(query_last_supply)
        df.at[idx, 'last_supply'] = str(last_supply[0]['m']['Name'] + "->" + last_supply[0]['n']['Name'])

        df.at[idx, 'is_chain_transaction'] = "True"
        df.at[idx, 'is_chain_transaction_comment'] = error_text

        if tr_name == last_enterprise_name:
            df.at[idx, 'moved_supply'] = "Last supply \n" + str(last_supply)
            start = last_supply[0]['m']['Name']
            ziel = last_supply[0]['n']['Name']

        elif tr_name == first_enterprise_name:
            df.at[idx, 'moved_supply'] = "First supply \n" + str(first_supply)
            start = first_supply[0]['m']['Name']
            ziel = first_supply[0]['n']['Name']

        else:
            query_zwischen_umsatz = f"""
                                OPTIONAL MATCH (n:Unternehmen)-[:BESTELLUNG]->(m:Unternehmen {{Name: "{tr_name}"}})
                                RETURN n, 'BESTELLUNG' as Info, m"""
            intermediate_supply = graph.query(query_zwischen_umsatz)
            # df.at[idx, 'zwischen_umsatz'] = str(zwischen_umsatz)
            # df.at[idx, 'zwischen_umsatz'] = df.at[idx, 'zwischen_umsatz'].replace("'", '"')

            query_proceeding_supply = f"""
                                                OPTIONAL MATCH (n:Unternehmen {{Name: "{tr_name}"}})-[:BESTELLUNG]->(m:Unternehmen)
                                                RETURN n, 'BESTELLUNG' as Info, m"""

            proceeding_supply = graph.query(query_proceeding_supply)

            dispatch_country = graph.query(query_find_dispatch_country)
            dispatch_country = dispatch_country[0]['n.Sitz']
            tv_uid = transport_responsibility[0]['n']['UID'][:2]
 
            if tv_uid == dispatch_country:
                start = intermediate_supply[0]['m']['Name']
                ziel = intermediate_supply[0]['n']['Name']
            else:
                start = proceeding_supply[0]['m']['Name']
                ziel = proceeding_supply[0]['n']['Name']


        query_movable_supply = (f'\nMATCH (a:Unternehmen {{Name: "{start}"}}),'
                                   f'(b:Unternehmen {{Name: "{ziel}"}})'
                                   '\nCREATE (a)-[:BEWEGTE_LIEFERUNG]->(b)')
        df.at[idx, 'cypher_statements'] = (df.at[idx, 'cypher_statements']
                                                     + ';\n'
                                                     + query_movable_supply)
        ################################################
        #Add the movable supply to the knowledge graph #
        ################################################
        graph.query(query_movable_supply)
        df.at[idx, 'identified_movable_supply'] = f"{start}->{ziel}"

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

    query_find_nodes = """
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

    result = graph.query(query_find_nodes)
    for record in result:
        if 'Transportverantwortung' in record["label"]:
            continue
        node_id = record["id"]
        label = record["label"]
        nodes[node_id] = {"label": label}

    # print(nodes)

    edges = []

    query_find_edges = """
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

    result = graph.query(query_find_edges)
    for record in result:
        if record["label"] == "HAT":
            continue
        source = record["source"]
        target = record["target"]
        label = record["label"]
        edges.append((source, target, label))


    # define graph layout
    dot = Digraph(format='pdf')
    dot.attr(rankdir='LR', size='10')  # Horizontal (Left-to-Right)

    # add nodes
    for node_id, props in nodes.items():

        if node_id == tv_name:
            dot.node(node_id, label=props['label'], shape='box', color='red', style='filled', fillcolor='lightblue')
        else:
            dot.node(node_id, label=props['label'], shape='box', style='filled', fillcolor='lightblue')

    # add edges
    for src, dst, label in edges:
        if 'BEWEGTE_LIEFERUNG' in label:
            dot.edge(src, dst, label=label, color='red', fontcolor='red')
        else:
            dot.edge(src, dst, label=label)

    # save as PDF
    dot.render(f"pdf_graphs/{id_internal}_{example_name}_visual_repr", format='pdf', cleanup=True)

    print(f"visualisation done, id {id_internal}_{example_name}")

########################################
#Queries to extract required knowledge #
########################################
query_last_enterprise = """
OPTIONAL MATCH (n:Unternehmen)-[:BESTELLUNG]->() 
WHERE NOT n:Transportverantwortung AND NOT EXISTS { MATCH ()-[:BESTELLUNG]->(n) } 

RETURN COALESCE(n, "Inconsistent, no solution.") AS result
"""
query_first_enterprise = """
OPTIONAL MATCH ()-[:BESTELLUNG]->(n:Unternehmen)
WHERE NOT n:Transportverantwortung AND NOT EXISTS { MATCH (n)-[:BESTELLUNG]->() } 

RETURN COALESCE(n, "Inconsistent, no solution.") AS result"""
query_find_transport_of_goods = """
    OPTIONAL MATCH (n:Unternehmen)-[:WARENBEWEGUNG]->(m:Unternehmen) 
    RETURN n, 'WARENBEWEGUNG' as Info, m
    """
query_find_dispatch_country = """
    OPTIONAL MATCH (n:Unternehmen)-[:WARENBEWEGUNG]->(m:Unternehmen) 
    RETURN n.Sitz
    """
query_transport_responsibility = """
    MATCH (n:Unternehmen)-[:HAT]->(:Transportverantwortung)
    RETURN n"""
query_no_of_products = """
    MATCH ()-[r]->()
    WHERE type(r) IN ['BESTELLUNG', 'WARENBEWEGUNG']
    RETURN count(DISTINCT r.Produkt) AS no_of_products"""
query_no_of_orders = "MATCH (:Unternehmen)-[r:BESTELLUNG]->(:Unternehmen) RETURN count(r) AS no_of_orders"
query_no_of_enterprises = "MATCH (n:Unternehmen) RETURN count(n) AS no_of_enterprises"
query_no_of_transports_of_goods = "MATCH (:Unternehmen)-[r:WARENBEWEGUNG]->(:Unternehmen) RETURN count(r) AS no_of_transports_of_goods"
query_no_of_tr = "MATCH (:Unternehmen)-[r:HAT]-(:Transportverantwortung) RETURN count(r) AS no_of_tr"


##################################################################################
# Load data and define data source                                               #
# Copy paste from this list of files containing the LLM output for each data set #
#                                                                                #
# EXAM.csv                                                                       #
# REALWORLD.csv                                                                  #
# DUPONT.csv                                                                     #
# MuW.csv                                                                        #
# KOLLMANN.csv                                                                   #
##################################################################################

filename = ("KOLLMANN.csv")
df_langsmith = pd.read_csv("data/" + filename)
reference_filename = ("sample_solutions_censored.xlsx")
df_database = pd.read_excel("data/" + reference_filename)
df_merged = df_langsmith.merge(df_database[['id','internal_id', 'sample_solution_movable_supply', 'data_set', 'name']], how="left",on='id')
df_langsmith = df_merged.copy()


#####################################################################
#Initialize KG DB and establish connection, instance must be running#
#####################################################################
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
        data_set = df_langsmith.at[idx, 'data_set']
        name = df_langsmith.at[idx, 'name']
        example_name = f"{data_set}_{name}"
        id_internal = df_langsmith.at[idx, 'internal_id']

        cypher_statement = data['Cypher Anweisungen']
        cypher_statement = clean_statement(cypher_statement)

        df_langsmith.at[idx, 'cypher_statements'] = cypher_statement
        graph_data = build_graph(cypher_statement)

        #Apply logic based rules#
        apply_logic_based_rules(graph, df_langsmith)


        #Check if result equals the sample solution#
        #Add 'Prüfung' if check is not successful  #
        if df_langsmith.at[idx, 'identified_movable_supply'] == df_langsmith.at[idx, 'sample_solution_movable_supply']:
            df_langsmith.at[idx, 'result'] = "correct"
        else:
            df_langsmith.at[idx, 'result'] = "check manually"

        #Generate visualization#
        #visualize_graph(graph, tr_name, id_internal, example_name)

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
df_langsmith.to_excel(f'output/{filename}_{timestamp}_df_with_query_outputs.xlsx', index=True)

