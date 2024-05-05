from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
# from langchain.chat_models import ChatOpenAI
from backend.llama2 import SaladChatOllama,MODEL

# from langchain.embeddings.openai import OpenAIEmbeddings
from backend.llama2 import SaladOllamaEmbeddings
from langchain.vectorstores import Chroma
import settings
import uuid
import os
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

from langchain.prompts import PromptTemplate

# template = """Use the following pieces of context to answer the question at the end. 
# If you don't know the answer, just say that you don't know, don't try to make up an answer. 
# Use three sentences maximum. Keep the answer as concise as possible. Be polite, and use emojis in your answers.
# {context}
# Question: {question}
# Helpful Answer:"""


template = """You are a finance executive.
Use the following pieces of context to answer the question at the end. 
If you don't know the answer, just say that you don't know, don't try to make up an answer. 
Use three sentences maximum. Keep the answer as concise as possible. 
Be polite, and use emojis in your answers.

Make sure your answers are factually correct and not estimated based on similar terms.
You must not to addition, subtraction in two numbers to get the third number (i.e. to get profit subtracting cost from revenue or to get EBIDTA adding taxes or interest, etc.). 
Understand the difference between terms like gross margin, net profit, revenue, net margin, and so on.
E.g., if you are asked about the net profit you must only answer if it's there. 

If Question is greeting don't consisder context

{context}
Question: {question}
Helpful Answer:"""


QA_CHAIN_PROMPT = PromptTemplate.from_template(template)


class PDFIndexer:
    def __init__(self, bfile) -> None:
        self.bfile = bfile
        self.uid = uuid.uuid4().__str__()

    def filehandler(self, bfile):
        filepath = f"docs/{self.uid}/{self.uid}.pdf"

        os.system(f"mkdir docs/{self.uid}")
        with open(f"docs/{self.uid}/{self.uid}.pdf", "wb") as f:
            f.write(bfile.file.read())

        return filepath

    def index(self):
        self.filepath = self.filehandler(bfile=self.bfile)

        loader = PyPDFLoader(self.filepath)
        pages = loader.load()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=450, chunk_overlap=0, separators=["\n\n", "\n", " ", ""]
        )

        documents = splitter.split_documents(pages)

        persist_directory = f"docs/{self.uid}/chroma"
        embedding = SaladOllamaEmbeddings(model=MODEL)

        vdb = Chroma(
            embedding_function=embedding,
            persist_directory=persist_directory,
        )
        

        for index,document in enumerate(documents):
            vdb.add_documents(documents=[document], embedding=embedding)
            vdb.persist()
            logging.info(f"Logs for document {index + 1}: {document.metadata}")
        

        return vdb


class Bot:
    def __init__(self, vdb) -> None:
        self.vdb = vdb
        self.qa = self._qa

    @property
    def _llm(self):
        llm_name = MODEL
        llm = SaladChatOllama(model_name=llm_name, temperature=0)
        return llm

    @property
    def _qa(self):
        memory = ConversationBufferMemory(
            memory_key="chat_history", return_messages=True
        )
        memory.input_key = "question"
        memory.output_key = "answer"
        retriever = self.vdb.as_retriever(score_threshold=.4)

        
        qa = ConversationalRetrievalChain.from_llm(
            self._llm,
            retriever=retriever,
            memory=memory,
            combine_docs_chain_kwargs={"prompt": QA_CHAIN_PROMPT},
            return_source_documents=True,
        )
        return qa

    def reply(self, question: str):
        result = self.qa({"question": question})

        if result["source_documents"]:
            sources = [
                str([result["source_documents"][i].metadata["page"] + 1]) for i in range(2)
            ]
        else:
            sources = ""

        return (
            result["answer"]
            + f"<br>🔗<span style='color:1DB100;font-size:12px'>{' '.join(sources)}</span>"
        )
    
    