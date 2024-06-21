from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores.faiss import FAISS
from langchain_community.llms import Ollama

from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
import warnings

# Ignore all warnings
warnings.filterwarnings("ignore")



### Loading the PDF through its Path
def load_pdf(path):
    loader = PyMuPDFLoader(path)
    return loader.load() 


### Dividing the PDF Contents into Chunks
def split_pdf(document):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=3000,        # creating chunks of 5000 characters
        chunk_overlap=200,      # an overlap of 200 characters to preserve context 
    )
    return text_splitter.split_documents(document)      # Splitting into documents with metadata 


### Specifying an Embedding Function
def get_embeddings():
    embeddings = HuggingFaceEmbeddings(model_name = "BAAI/bge-large-en-v1.5")   # Using the BGE model for embeddings
    return embeddings



### Creating a Vector Database to Store Embeddings
def vector_database(chunk_list):

    db=FAISS.from_documents(documents = chunk_list, embedding = get_embeddings())
    return db


### Querying the Database
def rag_query(chunk_list, query):
    llm = Ollama(model='llama3', temperature=0.7)
    db = vector_database(chunk_list)

    # Creating a prompt template for the multi-query retriever
    query_prompt = PromptTemplate(
        input_variable=['question'],
        template="""You are an AI language model assistant. Your taks is to generate five different versions of the given user question to retrieve relevant documents from a vector database. By generating multiple perspectives on the user question, your goal is to help the user overcome some of the limitations of the euclidean distance based similarity search. Provide these alternative questions separated by newlines.

        Original question: {question}"""
    )

    # Creating a multi-query retriever
    retriever = MultiQueryRetriever.from_llm(
        llm=llm,
        retriever=db.as_retriever(),
        prompt=query_prompt 
    )

    # Creating a prompt template for the chat prompt
    template = """Answer the question based on the following context. Also remember the history of previous answers, so if you are asked about anything regarding the previous chat, you can use that information to answer the question as well.
    Context: {context}
    Question: {question}"""
    prompt = ChatPromptTemplate.from_template(template=template)

    # Creating a chain of operations
    chain = (
    {'context': retriever, 'question': RunnablePassthrough()}       #passing the question as is using the RunnablePassthrough func
    | prompt                                                        #passing the context and question to the chat prompt template
    | llm
    | StrOutputParser()                                             #parsing the output as a string
)

    # Invoking the chain with the user query
    response = chain.invoke({'question': query})

    print('\n\n',response,'\n\n')
    

def main():
    file_path = input("Enter the pdf file path")
    # file_path = input("Enter the path of the PDF file: ")
    pdf = load_pdf(file_path)
    chunk_list = split_pdf(pdf)
    
    while True:
        query = input("Ask your question (Type 'exit' to exit...): ")
        if query.lower() == "exit" :
            print("\n\nThank you for using the service!!")
            exit()
        rag_query(chunk_list=chunk_list, query=query)

if __name__ == '__main__':
    main()