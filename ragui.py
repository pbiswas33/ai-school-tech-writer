import sys
from langsmith import traceable
from langchain_openai import ChatOpenAI
from langchain.prompts.prompt import PromptTemplate
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_pinecone import PineconeVectorStore
from langchain_community.document_loaders.pdf import PyPDFLoader
from PyQt5.QtCore import QThread, pyqtSignal
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QLineEdit, QPushButton, QMessageBox, QVBoxLayout, QFileDialog, QHBoxLayout, QTextEdit 
import os

os.environ['OPENAI_API_KEY'] = "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"
os.environ['PINECONE_API_KEY'] = "7d6144c6-b51c-4403-a580-7e565f87c2a3"
os.environ['LANGCHAIN_TRACING_V2'] = "true"
os.environ['LANGCHAIN_API_KEY'] = "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"

# This class handles the asynchronous generation of responses from the AI model.
# It inherits from QThread to run in a separate thread to prevent UI freezing.
class ResponseThread(QThread):
    # Signal to emit the response from the AI
    response_signal = pyqtSignal(str)

    def __init__(self, prompt, parent=None):
        # Initialize the QThread
        super(ResponseThread, self).__init__(parent)
        # Store the prompt provided by the user
        self.prompt = prompt

    def run(self):
        # Initialize the embeddings model with OpenAI's text-embedding model
        embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
        # Create a vector store for documents using Pinecone with a specific index name
        document_vectorstore = PineconeVectorStore(index_name="pdf-dev-textbook", embedding=embeddings)
        # Convert the document vector store into a retriever to fetch relevant documents
        retriever = document_vectorstore.as_retriever()
        # Retrieve documents relevant to the user's prompt
        context = retriever.get_relevant_documents(self.prompt)
        # Create a prompt template that includes the user's query and the retrieved context
        template = PromptTemplate(template="{query} Context: {context}", input_variables=["query", "context"])
        # Fill the template with the actual query and context
        prompt_with_context = template.invoke({"query": self.prompt, "context": context})
        # Initialize the language model with a specific temperature setting
        llm = ChatOpenAI(temperature=0.7)
        # Invoke the language model with the prompt that includes context
        results = llm.invoke(prompt_with_context)
        # Emit the content received from the language model as a signal
        self.response_signal.emit(results.content)

class FileUploadThread(QThread):
    # Signal to notify when file upload is complete
    upload_complete_signal = pyqtSignal(str)

    def __init__(self, file_path, parent=None):
        # Initialize the QThread with the parent if provided
        super(FileUploadThread, self).__init__(parent)
        # Store the file path to be processed
        self.file_path = file_path

    def run(self):
        # Initialize the PDF loader with the provided file path
        loader = PyPDFLoader(self.file_path)
        # Load the raw documents from the PDF file
        raw_docs = loader.load()
        # Initialize the text splitter with specified chunk size and overlap
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        # Split the loaded documents into manageable chunks
        documents = text_splitter.split_documents(raw_docs)
        # Initialize embeddings using OpenAI's text-embedding model
        embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
        # Create a Pinecone vector store from the documents with the specified embeddings and index name
        PineconeVectorStore.from_documents(documents=documents, embedding=embeddings, index_name="pdf-dev-textbook")
        # Emit a signal indicating the number of documents loaded and processed
        self.upload_complete_signal.emit(f"Loaded {len(documents)} splitted documents into the vector store.")
           
class RAGUIApp(QWidget):
    def __init__(self):
        # Initialize the QWidget base class
        super().__init__()
        # Set up the user interface
        self.init_ui()

    def init_ui(self):
        # Initialize label for file upload section
        self.upload_label = QLabel("Upload Knowledgebase:")
        # General label used for displaying status messages
        self.general_label = QLabel(":")
        # Button to trigger the file browsing dialog
        self.upload_button = QPushButton("Browse and Upload", self)
        # Entry field to display the selected file path
        self.file_path_entry = QLineEdit(self)
        # Connect the 'Browse and Upload' button to the file browsing method
        self.upload_button.clicked.connect(self.browse_file)
        
        # Input field for user to type their prompt
        self.prompt_input = QLineEdit(self)
        self.prompt_input.setPlaceholderText("Type your prompt here...")
        # Label to display the AI's response
        self.response_label = QLabel("AI Response:")
        # Text area to show the AI's response, set to read-only
        self.response_output = QTextEdit(self)
        self.response_output.setReadOnly(True)
        # Button to trigger the generation of the AI's response
        self.submit_button = QPushButton("Generate Response", self)
        # Connect the 'Generate Response' button to the response generation method
        self.submit_button.clicked.connect(self.generate_response)

        # Layout to manage the file upload widgets
        file_upload_layout = QHBoxLayout()
        file_upload_layout.addWidget(self.upload_label)
        file_upload_layout.addWidget(self.file_path_entry)

        # Layout for displaying general status messages
        general_label_layout = QVBoxLayout()
        general_label_layout.addWidget(self.general_label)
        
        # Layout for the prompt input field
        input_layout = QHBoxLayout()
        input_layout.addWidget(self.prompt_input)

        # Layout for the response generation button
        button_layout = QHBoxLayout()
        button_layout.addStretch(1)
        button_layout.addWidget(self.submit_button)
        
        # Main layout to arrange all sub-layouts vertically
        main_layout  = QVBoxLayout()
        
        # Layout for the upload button
        upload_button_layout  = QHBoxLayout()
        upload_button_layout.addStretch(1)
        upload_button_layout.addWidget(self.upload_button)
          
        # Add all sub-layouts to the main layout
        main_layout.addLayout(file_upload_layout)
        main_layout.addLayout(upload_button_layout)
        main_layout.addLayout(general_label_layout)
        main_layout.addLayout(input_layout)
        main_layout.addLayout(button_layout)
        main_layout.addWidget(self.response_label)
        main_layout.addWidget(self.response_output)

        # Set the main layout for the widget
        self.setLayout(main_layout)

        # Set window title and geometry
        self.setWindowTitle("RAG UI")
        self.setGeometry(100, 100, 800, 600)  # Adjusted height to accommodate new elements
   
    def browse_file(self):
        """
        Opens a file dialog for the user to select a PDF file and sets the file path in the text entry.
        If a file is selected, it triggers the file upload process.
        """
        file_name, _ = QFileDialog.getOpenFileName(self, "Open File", "", "PDF Files (*.pdf)")
        if file_name:
            self.file_path_entry.setText(file_name)  # Display the selected file path in the text entry
            self.upload_file(file_name)  # Initiate the upload process for the selected file
            
    def upload_file(self, file_path):
        """
        Starts the file upload process in a separate thread and updates the UI to show the uploading status.
        """
        self.general_label.setText("Uploading and processing file...")  # Update the UI with the status
        self.file_upload_thread = FileUploadThread(file_path)  # Create a thread to handle file upload
        self.file_upload_thread.upload_complete_signal.connect(self.update_upload_status)  # Connect the signal for upload completion to the handler
        self.file_upload_thread.start()  # Start the thread

    def update_upload_status(self, message):
        """
        Updates the general label with the status message received from the upload thread.
        """
        self.general_label.setText(message)  # Update the UI with the new status message
        
    def generate_response(self):
        """
        Initiates the generation of a response based on the user's input in a separate thread and updates the UI to indicate that the response is being generated.
        """
        self.response_output.setText("Generating response...")  # Indicate that the response generation has started
        prompt = self.prompt_input.text().strip()  # Get the user's input, stripping any leading/trailing whitespace
        self.response_thread = ResponseThread(prompt)  # Create a thread to generate the response
        self.response_thread.response_signal.connect(self.update_response)  # Connect the signal for response completion to the handler
        self.response_thread.start()  # Start the thread

    def update_response(self, content):
        """
        Updates the response output area with the content received from the response generation thread.
        """
        self.response_output.setPlainText(content)  # Display the generated response in the text area

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = RAGUIApp()
    window.show()
    sys.exit(app.exec_())