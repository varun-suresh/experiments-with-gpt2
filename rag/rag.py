# Encode the documents using the embedding_model - i.e For each sentence (of sentence_size) in the document, create an embedding. Return the embeddings
import numpy as np
import faiss
class RAG:
    def __init__(self,embedding_model, embedding_model_size,generate_model,sentence_size,overlap_size,k):
        self.embedding_model = embedding_model
        self.generate_model = generate_model
        self.sentence_size = sentence_size
        self.overlap_size = overlap_size
        self.output_text = []
        self.index = faiss.IndexFlatL2(embedding_model_size)
        self.k = k

    def add_to_knowledge_base(self,text_docs):
        doc_embeddings = []
        for doc in text_docs: 
            doc_embedding, doc_text = self.embedding_model.encode(open(doc).read(),
                                                                self.sentence_size,
                                                                self.overlap_size)
            doc_embeddings.append(doc_embedding)
            self.output_text.extend(doc_text)

        self.index.add(np.vstack(doc_embeddings))

    # Get the top k documents most relevant to query
    def retrieve(self,query,k):
        # For now assuming that the query length is one or two sentences, less than the sentence_size
        query_embedding, _ = self.embedding_model.encode(query,self.sentence_size,self.overlap_size)
        dist, ann = self.index.search(query_embedding,k)

        relevant_docs = []
        for neighbor in ann[0]:
            relevant_docs.append(self.output_text[neighbor])
        
        return relevant_docs
    
    def _generate(self,query,relevant_docs,max_new_tokens):
        context = " ".join(doc for doc in relevant_docs)
        input_text = f"Context:{context} Query: {query} Response:"
        response = self.generate_model.generate(input_text,max_new_tokens)
        return f"Query: {query} Response: {response[len(input_text):]}"

    def get_response(self,query,max_new_tokens=50):
        relevant_docs = self.retrieve(query=query,k=self.k)
        return self._generate(query,relevant_docs,max_new_tokens)
