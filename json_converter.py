import json
from langchain_community.graphs.graph_document import GraphDocument



# Dein JSON-LD-Dokument
json_ld = {
    "@context": "https://schema.org",
    "@graph": [
        {"@type": "Company", "name": "Ö3", "country": "Austria", "VAT number": "AT12345678"},
        {"@type": "Company", "name": "Ö2", "country": "Austria", "VAT number": "AT87654321"},
        {"@type": "Company", "name": "Ö1", "country": "Austria", "VAT number": "AT11223344"},
        {"@type": "Company", "name": "CH", "country": "Switzerland", "VAT number": "CHE99887766"},
        {"@type": "Transport", "name": "Transport organized by Ö3"}
    ],
    "relationships": [
        {"@type": "ORDERS_FROM", "source": {"@id": "Ö3"}, "target": {"@id": "Ö2"}, "date": "2023-01-01", "transported_good": "Machine"},
        {"@type": "ORDERS_FROM", "source": {"@id": "Ö2"}, "target": {"@id": "Ö1"}, "date": "2023-01-01", "transported_good": "Machine"},
        {"@type": "ORDERS_FROM", "source": {"@id": "Ö1"}, "target": {"@id": "CH"}, "date": "2023-01-01", "transported_good": "Machine"},
        {"@type": "RESPONSIBLE_FOR", "source": {"@id": "Ö3"}, "target": {"@id": "Transport organized by Ö3"}}
    ]
}

# Umwandlung in GraphDocument
nodes = []
node_lookup = {}

# Erstelle die Nodes
for entry in json_ld["@graph"]:
    node = {
        "id": entry["name"],
        "type": entry["@type"],
        "properties": {k: v for k, v in entry.items() if k not in ["@type", "name"]}
    }
    nodes.append(node)
    node_lookup[entry["name"]] = node

# Erstelle die Kanten (Relationships)
relationships = []
for rel in json_ld["relationships"]:
    relationships.append({
        "source": node_lookup[rel["source"]["@id"]],
        "target": node_lookup[rel["target"]["@id"]],
        "type": rel["@type"],
        "properties": {k: v for k, v in rel.items() if k not in ["@type", "source", "target"]}
    })

# Erstelle das GraphDocument
graph_doc = GraphDocument(nodes=nodes, relationships=relationships)

# Ausgabe
print(graph_doc)
