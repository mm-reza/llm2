import os
from dotenv import load_dotenv

from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import CSVLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceInstructEmbeddings
from langchain_community.llms import GooglePalm

# Load environment variables (e.g., API keys)
load_dotenv()

# Initialize the language model
llm = GooglePalm(
    google_api_key=os.getenv("GOOGLE_API_KEY"),
    temperature=0.2
)

# Initialize HuggingFace Instructor embeddings
instructor_embeddings = HuggingFaceInstructEmbeddings(
    model_name="hkunlp/instructor-xl",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True}
)

# Path for storing FAISS index
VECTOR_DB_PATH = "faiss_index"


def create_vector_db(csv_file: str = "codebasics_faqs.csv") -> None:
    """
    Loads data from a CSV file and creates a FAISS vector database.
    Saves the FAISS index locally.
    """
    print("Updating LLM knowledge...")

    # Load FAQ data from CSV
    loader = CSVLoader(file_path=csv_file, source_column="prompt", encoding="cp1252")
    data = loader.load()

    # Create FAISS vector database
    vectordb = FAISS.from_documents(documents=data, embedding=instructor_embeddings)

    # Save vector database locally
    vectordb.save_local(VECTOR_DB_PATH)
    print("Database update completed.")


def load_vector_db() -> FAISS:
    """
    Loads the FAISS vector database from the local directory.
    Returns the FAISS instance.
    """
    return FAISS.load_local(VECTOR_DB_PATH, instructor_embeddings)


def get_qa_chain() -> RetrievalQA:
    """
    Loads the FAISS vector database and sets up a retrieval-based QA chain.
    Returns the RetrievalQA instance.
    """

    vectordb = load_vector_db()

    # Create a retriever with fuzzy search enabled
    retriever = vectordb.as_retriever(
        search_type="similarity",  # Enables fuzzy matching
        search_kwargs={"k": 5}  # Retrieves top 5 similar results
    )

    # Define prompt template
    prompt_template = """Given the following context and a question, generate an answer based on this context only.
    In the answer, try to provide as much text as possible from the "response" section in the source document context without making major changes.
    If the answer is not found in the context, generate a possible answer from the LLM in the second paragraph.
    Provide a reference list of source link(s) in the third paragraph.

    CONTEXT: {context}

    QUESTION: {question}"""

    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

    # Create QA chain
    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt},
    )


if __name__ == "__main__":
    create_vector_db()
    qa_chain = get_qa_chain()

    # Sample query to test
    query = "Do you have a JavaScript course?"
    response = qa_chain.invoke({"query": query})

    print(response)
