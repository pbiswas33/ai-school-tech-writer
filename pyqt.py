import sys
from langsmith import traceable
from langchain_openai import ChatOpenAI
from langchain.prompts.prompt import PromptTemplate
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_pinecone import PineconeVectorStore
from langchain_community.document_loaders.pdf import PyPDFLoader
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QLineEdit, QPushButton, QMessageBox, QVBoxLayout, QFileDialog, QHBoxLayout,QTextEdit 
import os



class SampleApp(QWidget):
    def __init__(self):
        super().__init__()
        self.init_ui()

    def init_ui(self):
        # Create Widgets for file upload
        self.upload_label = QLabel("Select a file to upload:")
        self.general_label = QLabel(":")
        self.upload_button = QPushButton("Browse", self)
        self.file_path_entry = QLineEdit(self)
        self.upload_button.clicked.connect(self.browse_file)
        
        self.prompt_label = QLabel("Enter your prompt:")
        self.prompt_input = QLineEdit(self)
        self.prompt_input.setPlaceholderText("Type your prompt here...")
        self.response_label = QLabel("AI Response:")
        self.response_output = QTextEdit(self)
        self.response_output.setReadOnly(True)
        self.submit_button = QPushButton("Generate Response", self)
        self.submit_button.clicked.connect(self.generate_response)

        file_upload_layout = QHBoxLayout()
        file_upload_layout.addWidget(self.upload_label)
        file_upload_layout.addWidget(self.file_path_entry)

        input_layout = QHBoxLayout()
        input_layout.addWidget(self.prompt_label)
        input_layout.addWidget(self.prompt_input)

        button_layout = QHBoxLayout()
        button_layout.addStretch(1)
        button_layout.addWidget(self.submit_button)
        
        # Set up the layout
        main_layout  = QVBoxLayout()
        
        upload_button_layout  = QHBoxLayout()
        upload_button_layout.addStretch(1)
        upload_button_layout.addWidget(self.upload_button)
            
        ##layout.addWidget(self.general_label)
        
        main_layout.addLayout(file_upload_layout)
        main_layout.addLayout(upload_button_layout)
        main_layout.addLayout(input_layout)
        main_layout.addLayout(button_layout)
        main_layout.addWidget(self.response_label)
        main_layout.addWidget(self.response_output)

        self.setLayout(main_layout)

        # Set window properties
        self.setWindowTitle("RAG UI")
        self.setGeometry(100, 100, 800, 600)  # Adjusted height to accommodate new elements
   
    def browse_file(self):
        # Open file dialog to select a file
        file_name, _ = QFileDialog.getOpenFileName(self, "Open File", "", "PDF Files (*.pdf)")
        if file_name:
            self.file_path_entry.setText(file_name)
            self.upload_file(file_name)
    def upload_file(self, file_path):
        # Here you can add the logic to handle the file upload or processing
        self.general_label.setText(f"You have selected: {file_path}")
        loader = PyPDFLoader(file_path)
        raw_docs = loader.load()

        # Split documents into smaller chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        documents = text_splitter.split_documents(raw_docs)
        self.general_label.setText(f"Going to add {len(documents)} documents to Pinecone")
        # Choose the embedding model and vector store
        embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
        PineconeVectorStore.from_documents(documents=documents, embedding=embeddings, index_name="pdf-dev-textbook")
        self.general_label.setText("Loading to vectorstore done")
        
    def generate_response(self):
        self.response_output.setText("Generating response...")
        prompt = self.prompt_input.text().strip()
        embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
        # Querying the vector database for "relevant" docs
        document_vectorstore = PineconeVectorStore(index_name="pdf-dev-textbook", embedding=embeddings)
        retriever = document_vectorstore.as_retriever()
        context = retriever.get_relevant_documents(prompt)
        # Adding context to our prompt
        template = PromptTemplate(template="{query} Context: {context}", input_variables=["query", "context"])
        prompt_with_context = template.invoke({"query": prompt, "context": context})
        # Asking the LLM for a response from our prompt with the provided context
        llm = ChatOpenAI(temperature=0.7)
               
        results = llm.invoke(prompt_with_context)
        print(results.content)
        self.response_output.setPlainText(results.content)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = SampleApp()
    window.show()
    sys.exit(app.exec_())
