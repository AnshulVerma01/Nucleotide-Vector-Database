import argparse
import json
import numpy as np
import faiss
import os
import torch
from transformers import AutoTokenizer, AutoModel, BertConfig
from Bio import SeqIO

class VectorDatabase:
    def __init__(self, model_name='zhihan1996/DNABERT-2-117M', storage_path='vector_db/'):
        # Initialize the tokenizer and model for DNABERT
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True, config=BertConfig.from_pretrained(model_name))
        self.storage_path = storage_path
        self.embeddings = None
        self.metadata = []
        self.sequences = []  # Store sequences for retrieval
        self.index = None

        # Create storage directory if it does not exist
        os.makedirs(self.storage_path, exist_ok=True)

        # Try to load existing database if available
        self.load()

    def _encode(self, sequence):
        """Encode a DNA sequence into an embedding."""
        inputs = self.tokenizer(sequence, return_tensors='pt')["input_ids"]
        with torch.no_grad():
            hidden_states = self.model(inputs)[0]  # [1, sequence_length, 768]
        
        # Compute embeddings with mean pooling
        embedding_mean = torch.mean(hidden_states[0], dim=0).numpy()
        return embedding_mean

    def create(self, file_path, metadata=None):
        """Create embeddings for the DNA sequences in a FASTA file and add them to the database."""
        sequences = [str(record.seq) for record in SeqIO.parse(file_path, 'fasta')]

        embeddings = [self._encode(seq) for seq in sequences]
        self.metadata.extend([metadata] * len(sequences))
        self.sequences.extend(sequences)  # Update self.sequences with the new sequences

        for embedding in embeddings:
            self._add_embedding(embedding)

        self._save()  # Automatically save changes
        print(f"Entries created for file: {file_path}")

    def _add_embedding(self, embedding):
        """Add an embedding to the FAISS index."""
        if self.embeddings is None:
            self.embeddings = np.array([embedding])
            d = embedding.shape[0]
            self.index = faiss.IndexFlatL2(d)
        else:
            self.embeddings = np.vstack((self.embeddings, embedding))

        self.index.add(np.array([embedding]))

    def read(self, query_file_path, top_k=5):
        """Search for similar entries in the database using the DNA content of a FASTA file."""
        if self.index is None:
            print("No index found. Please load the database.")
            return

        query_sequences = [str(record.seq) for record in SeqIO.parse(query_file_path, 'fasta')]

        # Iterate over each query sequence
        for query_dna in query_sequences:
            query_embedding = self._encode(query_dna)
            distances, indices = self.index.search(np.array([query_embedding]), top_k)
            
            # Collect results for each sequence
            results = []
            for j, i in enumerate(indices[0]):
                metadata = self.metadata[i]
                sequence = self.sequences[i]
                similarity_score = 1 - distances[0][j]
                results.append((metadata, sequence, similarity_score))

            # Print results for each query sequence
            print(f"Query Sequence: {query_dna}")
            if results:
                for metadata, sequence, similarity_score in results:
                    print(f"Metadata: {metadata}, Similar Sequence: {sequence}, Similarity Score: {similarity_score:.2f}")
            else:
                print("No similar sequences found.")
            print()  # Add an empty line between results of different query sequences

    def update(self, text_id, new_file_path, new_metadata=None):
        """Update an existing entry by inserting sequences from a new FASTA file at the specified position."""
        if text_id >= len(self.metadata):
            print("Invalid text ID.")
            return
    
        # Read new sequences from the FASTA file
        new_sequences = [str(record.seq) for record in SeqIO.parse(new_file_path, 'fasta')]
        
        if not new_sequences:
            print("No sequences found in the provided FASTA file.")
            return
    
        # Insert new sequences starting from the specified text_id
        self.sequences[text_id:text_id] = new_sequences  # Insert at the specified text_id
        self.metadata[text_id:text_id] = [new_metadata or {}] * len(new_sequences)  # Insert metadata for each new sequence
        
        # Update embeddings for new sequences
        for seq in new_sequences:
            new_embedding = self._encode(seq)
            self.embeddings = np.insert(self.embeddings, text_id, new_embedding, axis=0)
            text_id += 1  # Increment text_id for the next sequence
    
        # Rebuild the FAISS index
        self._rebuild_index()
        self._save()  # Automatically save changes
        print(f"Inserted {len(new_sequences)} sequences starting at position {text_id - len(new_sequences)}.")


    def delete(self, text_id):
        """Delete all entries associated with the metadata of the given ID."""
        if text_id >= len(self.metadata):
            print("Invalid text ID.")
            return

        metadata_to_delete = self.metadata[text_id]
        
        # Find indices of all entries with the same metadata
        indices_to_delete = [i for i, meta in enumerate(self.metadata) if meta == metadata_to_delete]

        # Remove all sequences and embeddings associated with the metadata
        self.embeddings = np.delete(self.embeddings, indices_to_delete, axis=0)
        self.metadata = [meta for i, meta in enumerate(self.metadata) if i not in indices_to_delete]
        self.sequences = [seq for i, seq in enumerate(self.sequences) if i not in indices_to_delete]

        # Rebuild the FAISS index
        self._rebuild_index()
        self._save()  # Automatically save changes
        print(f"All entries associated with metadata '{metadata_to_delete}' deleted.")

    def _rebuild_index(self):
        """Rebuild the FAISS index after an update or delete operation."""
        if self.embeddings is not None and len(self.embeddings) > 0:
            d = self.embeddings.shape[1]
            self.index = faiss.IndexFlatL2(d)
            self.index.add(self.embeddings)
        else:
            self.index = None

    def _save(self):
        """Save embeddings, metadata, and index to disk."""
        if self.embeddings is not None:
            np.save(os.path.join(self.storage_path, 'embeddings.npy'), self.embeddings)
        with open(os.path.join(self.storage_path, 'metadata.json'), 'w') as f:
            json.dump(self.metadata, f)
        if self.index is not None:
            faiss.write_index(self.index, os.path.join(self.storage_path, 'index.faiss'))
        with open(os.path.join(self.storage_path, 'sequences.json'), 'w') as f:
            json.dump(self.sequences, f)
        print("Database saved to disk.")

    def load(self):
        """Load embeddings, metadata, and index from disk."""
        embeddings_path = os.path.join(self.storage_path, 'embeddings.npy')
        metadata_path = os.path.join(self.storage_path, 'metadata.json')
        index_path = os.path.join(self.storage_path, 'index.faiss')
        sequences_path = os.path.join(self.storage_path, 'sequences.json')

        if os.path.exists(embeddings_path):
            self.embeddings = np.load(embeddings_path)
        else:
            self.embeddings = None

        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                self.metadata = json.load(f)
        else:
            self.metadata = []

        if os.path.exists(sequences_path):
            with open(sequences_path, 'r') as f:
                self.sequences = json.load(f)
        else:
            self.sequences = []

        if os.path.exists(index_path):
            self.index = faiss.read_index(index_path)
        else:
            self.index = None

        print("Database loaded from disk.")

def parse_args():
    parser = argparse.ArgumentParser(description='Vector Database CLI')
    subparsers = parser.add_subparsers(dest='command')

    # Subparser for 'create'
    parser_create = subparsers.add_parser('create', help='Create new entries from a FASTA file')
    parser_create.add_argument('file_path', type=str, help='Path to the FASTA file')
    parser_create.add_argument('--metadata', type=json.loads, default='{}', help='Metadata as JSON')

    # Subparser for 'read'
    parser_read = subparsers.add_parser('read', help='Read/search entries using a query FASTA file')
    parser_read.add_argument('query_file_path', type=str, help='Path to the query FASTA file')
    parser_read.add_argument('--top_k', type=int, default=5, help='Number of top results to return')

    # Subparser for 'update'
    parser_update = subparsers.add_parser('update', help='Update an existing entry using a new FASTA file')
    parser_update.add_argument('text_id', type=int, help='ID of the text to update')
    parser_update.add_argument('new_file_path', type=str, help='Path to the new FASTA file')
    parser_update.add_argument('--new_metadata', type=json.loads, default='{}', help='Updated metadata as JSON')

    # Subparser for 'delete'
    parser_delete = subparsers.add_parser('delete', help='Delete an entry')
    parser_delete.add_argument('text_id', type=int, help='ID of the entry to delete')

    # Subparser for 'load'
    parser_load = subparsers.add_parser('load', help='Load the database from disk')

    return parser.parse_args()

def main():
    args = parse_args()
    db = VectorDatabase()

    if args.command == 'create':
        db.create(args.file_path, args.metadata)
    elif args.command == 'read':
        db.read(args.query_file_path, args.top_k)
    elif args.command == 'update':
        db.update(args.text_id, args.new_file_path, args.new_metadata)
    elif args.command == 'delete':
        db.delete(args.text_id)
    elif args.command == 'load':
        db.load()
    else:
        print("Invalid command. Use 'create', 'read', 'update', 'delete', or 'load'.")

if __name__ == '__main__':
    main()

