!pip install langchain langchain-community langchain-core langchain-text-splitters
!pip install chromadb
!pip install sentence-transformers
!curl -fsSL https://ollama.ai/install.sh|sh
!nohup ollama serve >/dev/null 2>&1 &
import time
time.sleep(5) # Give Ollama some time to start
!ollama pull mistral
speech_text="""
The real remedy is to destroy the belief in the sanctity of the shastras. How do you expect to succeed if you allow the shastras to continue to be\nheld as sacred and infallible? You must take a stand against the scriptures. Either you must stop the practice of caste or you must stop believing\nin the shastras. You cannot have both. The problem of caste is not a problem of social reform. It is a problem of overthrowing the authority of the\nshastras. So long as people believe in the sanctity of the shastras, they will never be able to get rid of caste. The work of social reform is like the\nwork of a gardener who is constantly pruning the leaves and branches of a tree without ever attacking the roots. The real enemy is the belief in the\nshastras.
"""
with open("speech.txt","w") as f:
  f.write(speech_text)
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA
loader= TextLoader("speech.txt")
doc=loader.load()

langchain_text_splitters=CharacterTextSplitter(chunk_size=300,chunk_overlap=50)
chunks=langchain_text_splitters.split_documents(doc)
len(chunks)

for i, chunk in enumerate(chunks):
  print(f"--- Chunk {i+1} ---")
  print(chunk.page_content)
  print("\n"+"="*50+"\n")
embeddings=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore=Chroma.from_documents(chunks,embedding=embeddings,collection_name="ambedkar_rag")
retriever=vectorstore.as_retriever()
llm=Ollama(model="mistral")
qa=RetrievalQA.from_chain_type(llm=llm,chain_type="stuff",retriever=retriever)
summary_question = "What is the main argument of the speech and what is the real remedy suggested?"
result = qa.invoke(summary_question)
print(result["result"])

def ask():
   while True:
    q=input("Please ASK your question or type exit to end....:")
    if q.lower() in ["exit","quit","bye"]:
       print("goodbye!!!")
       break

    result=qa.invoke(q)
    print("\nAnswer is :","\n")
    print(result["result"])


ask()
