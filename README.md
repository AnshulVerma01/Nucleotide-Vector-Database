# Vector Database
This Python script provides a command-line interface (CLI) for managing a vector database of DNA sequences using DNABERT. The database allows for creating, reading, updating, and deleting DNA sequence entries stored as embeddings. It uses FAISS for efficient similarity search.


## Installation
You can install the dependencies using pip:
```Bash
pip install torch transformers faiss-cpu numpy biopython
```

## Create

Add a new text entry by converting its content into embeddings and storing it in the database.
```python
python seq_vdb.py create /content/cancer.fasta --metadata '{"Topic": "Breast cancer"}'
python seq_vdb.py create /content/non_coding.fasta --metadata '{"Topic": "Non coding variant"}'
python seq_vdb.py create /content/query.fasta --metadata '{"Topic": "Query"}'
```

## Read

Perform a similarity search using a query file and retrieve top k most similar entries.
```python
python seq_vdb.py read /content/query.fasta --top_k 3
```

## Update

Update the text content and metadata for an existing entry.
```python
python seq_vdb.py update 5 --new_metadata '{"Topic": "query"}' /content/query.fasta
```

## Delete

Delete an entry by its ID.
```python
python seq_vdb.py delete 5
```



