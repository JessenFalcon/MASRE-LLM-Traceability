# Pub-Test
The Dataset and Code used for Explainable Traceability

Environment：
Use requirements.txt to build running environment.

Dataset:
Unzip the compressed Dataset file "Datset.rar" and get the Requirement-Code Traceability ground truth with souce code.

Phase 1: Knowledge Acquision:
Config your API key and URL in Knowldegd_Build.py, input your project source code directory. After running, the Vector Embedding Retrieval Base, Metadata in json, and knowledge graph in json will be output.

Phase2: Explainable Reasoning:
Config your API key and URL in RCTracebility.py, input your project knowledge base and dataset address, run and get the traceability links with explainable report.
