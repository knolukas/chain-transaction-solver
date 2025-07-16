# Documentation

![20250708_architecture drawio](https://github.com/user-attachments/assets/279a6e6d-8b24-4a6f-b33b-4544b24923f5)

This application was developed to solve Chain Transaction (CT) Cases under Austrian tax law by the use of LLMs and Knowledge Graphs (KG).
To this purpose it extracts knowledge from given natural language text and generates a knowlege graph in Neo4j Aura, runs several queries and applies a set of logic-based rules to identify the movable supply of the CT.

To run this application you must install requirements.txt

First, it establishes a conntection to the Neo4j Aura database and Langsmith.
Make sure to use the correct credentials in an .env file.
Make sure the instance of Neo4j Aura is running.

There are 2 files in this repository:
run_experiment.py --> used to access the LLM via Langsmith and generate Cypher statements
ct_solver.py --> used to process the LLM output, generate a KG and identify the movable supply


## Functions   	 	
### apply_logic_based_rules(graph, df)
Application of the logic based rules.
First, run queries to identify and extract required data.
Second, check whether there is a valid chain transaction.
Third, identify the movable supply.
:param graph: Knowledge graph from the KG-database
:param df: Dataframe to store the output
:return: none

### build_graph(query)
Build the graph in Neo4j
:param query: Statements to build the graph
:return: Graph data

### clean_statement(statement)
Clean statement to remove inconsistencies
:param statement: Statement to clean

:return: Cleaned statement

### delete_graph()
Delete the graph from Neo4j
:return: none

### replace_name_fields(text)
Replace name field to remove inconsistencies
:param text: Text to clean
:return: cleaned text

### visualize_graph(graph, tv_name, id_internal, example_name)
Custom visualization of the graph
:param graph: Knowledge graph
:param tv_name: Name of transport responsible enterprise
:param id_internal: Internal ID of the case
:param example_name: Name of the case
:return: none

Next we define a set of queries to extract information from the KG:
## Queries

### Find last enterprise
`query_letztes_unternehmen = """
OPTIONAL MATCH (n:Unternehmen)-[:BESTELLUNG]->() 
WHERE NOT n:Transportverantwortung AND NOT EXISTS { MATCH ()-[:BESTELLUNG]->(n) } 

RETURN COALESCE(n, "Inconsistent, no solution.") AS result
"""`

### Find first enterprise
query_erstes_unternehmen = """
OPTIONAL MATCH ()-[:BESTELLUNG]->(n:Unternehmen)
WHERE NOT n:Transportverantwortung AND NOT EXISTS { MATCH (n)-[:BESTELLUNG]->() } 

RETURN COALESCE(n, "Inconsistent, no solution.") AS result"""

### Find transport
query_finde_warenbewegung = """
    //Finde Lieferung
    OPTIONAL MATCH (n:Unternehmen)-[:WARENBEWEGUNG]->(m:Unternehmen) 
    RETURN n, 'WARENBEWEGUNG' as Info, m
    """

### Find country where goods depart
query_finde_abgangsstaat = """
    //Finde Lieferung
    OPTIONAL MATCH (n:Unternehmen)-[:WARENBEWEGUNG]->(m:Unternehmen) 
    RETURN n.Sitz
    """

### Check if valid CT
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

### Find Transport responsibility:
query_transportverantwortung = """
    MATCH (n:Unternehmen)-[:HAT]->(:Transportverantwortung)
    RETURN n"""

### Identify total number of traded and distinct goods:
query_anzahl_produkte = """
    MATCH ()-[r]->()
    WHERE type(r) IN ['BESTELLUNG', 'WARENBEWEGUNG']
    RETURN count(DISTINCT r.Produkt) AS anzahl_unterschiedlicher_produkte"""


## Data processing

Next, the data must be loaded into the application. The only allowed dataformat is .csv. The data is generated from the Langsmith LLM application.
It is important that it included all required columns.
As a first step, the data is merged with the stored case database to join input and output by ID.

Next, the graph in initialized and the connection to the Neo4j Aura database is established.

In the following for-loop, the application performs all required actions row-by-row for the transmitted .csv file.

1. Load data and encode
2. Store data in pandas dataframe
3. Construct unique names for each case for storage
4. Run Cypher statements from the df to generate a KG in the database
5. Apply the set of logic based rules to the KG data
6. Run a quick check if the identified movable supply is equal to the sample solution. If it matches the solution write 'richtig' into the comparison column. Else write 'prüfung' to signal that a manual check is neccessary.
7. Visualize the graph including the movable supply


## Exception handling

We handle every exception by printing a set of available information including: exception type, exception filename, exception line number and exception message.
