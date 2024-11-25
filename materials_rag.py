from langchain_openai import ChatOpenAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.prompts import PromptTemplate
from pydantic import SecretStr
import logging
import os
from typing import AsyncGenerator, Dict, List, Optional, Any
from datetime import datetime
import uuid
import asyncio
from dotenv import load_dotenv
from config import OPENAI_API_KEY

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MaterialsRAGSystem:
    """
    A Retrieval-Augmented Generation (RAG) system specialized for construction materials information.

    This class implements a conversational AI system that processes and retrieves information about
    construction materials, technical specifications, building codes, and related documentation.
    It maintains conversation history and provides both synchronous and streaming query capabilities.

    Attributes:
        llm: ChatOpenAI instance for language model interactions
        embeddings: HuggingFaceEmbeddings instance for text embeddings
        text_splitter: CharacterTextSplitter for document chunking
        vector_store: Chroma vector store for document storage and retrieval
        qa_chain: ConversationalRetrievalChain for managing Q&A interactions
        conversations: Dictionary storing conversation histories by session ID
        session_metadata: Dictionary storing metadata for each conversation session

    Args:
        model_path (str): Path to the language model
        materials_data (Dict): Dictionary containing structured materials data including:
            - product_catalog: List of product information
            - technical_documents: List of technical documentation
            - building_codes: List of building code information
            - installation_guides: List of installation guides
            - safety_documents: List of safety documentation
            - material_alternatives: List of alternative material options

    Raises:
        ValueError: If materials_data format is invalid or no valid documents are found
        Exception: For initialization errors related to model loading or document processing

    Example:
        >>> materials_data = {
        ...     'product_catalog': [{
        ...         'name': 'Steel Beam X100',
        ...         'id': 'SB100',
        ...         'category': 'Structural Steel'
        ...     }]
        ... }
        >>> rag_system = MaterialsRAGSystem('path/to/model', materials_data)
        >>> response = rag_system.query("What are the specifications for Steel Beam X100?")
    """
    def __init__(self, model_path: str, materials_data: Dict):
        """
        Sets up the RAG system by initializing the language model, embeddings, and document storage.
        Also creates empty conversation storage for tracking chat history.
        """
        try:
            load_dotenv()
            os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

            self.llm = ChatOpenAI(
                api_key=SecretStr(OPENAI_API_KEY),
                model="gpt-4o-mini",
                temperature=0.01,
                max_tokens=1000,
                verbose=True
            )

            self.embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                cache_folder="./models"
            )

            self.text_splitter = CharacterTextSplitter(
                separator=" ",
                chunk_size=500,
                chunk_overlap=50,
                length_function=len
            )

            self.vector_store = self._initialize_vector_store(materials_data)
            self.qa_chain = self._initialize_qa_chain()
            self.conversations: Dict[str, List] = {}
            self.session_metadata: Dict[str, Dict] = {}

        except Exception as e:
            logger.error(f"Initialization error: {e}")
            raise

    def _initialize_vector_store(self, materials_data: Dict) -> Chroma:
        """
        Creates a searchable database of documents from the provided materials data.
        Converts text into vectors for efficient searching.
        """
        if not isinstance(materials_data, dict):
            raise ValueError("Invalid materials_data format")

        documents = self._prepare_documents(materials_data)
        splits = self.text_splitter.create_documents(documents)
        
        vector_store = Chroma.from_documents(
            documents=splits,
            embedding=self.embeddings,
            persist_directory="./chroma_db"
        )
        
        return vector_store

    def _initialize_qa_chain(self) -> ConversationalRetrievalChain:
        """
        Sets up the question-answering system with custom prompts and retrieval settings.
        This handles how questions are processed and answered using the stored documents.
        """
        custom_prompt = PromptTemplate(
            template=self._get_prompt_template(),
            input_variables=["chat_history", "context", "question"]
        )

        return ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=self.vector_store.as_retriever(search_kwargs={"k": 4}),
            return_source_documents=True,
            combine_docs_chain_kwargs={
                "prompt": custom_prompt,
                "document_prompt": PromptTemplate(
                    input_variables=["page_content"],
                    template="[DOC] {page_content}\n============================================\n"
                ),
            },
            verbose=True
        )

    def _prepare_documents(self, materials_data: Dict) -> List[str]:
        """
        Processes different types of materials data (products, technical docs, etc.)
        into a format that can be stored and searched efficiently.
        """
        document_processors = {
            'product_catalog': self._process_product,
            'technical_documents': self._process_technical_doc,
            'building_codes': self._process_building_code,
            'installation_guides': self._process_installation_guide,
            'safety_documents': self._process_safety_doc,
            'material_alternatives': self._process_material_alternative
        }

        documents = []
        for data_type, processor in document_processors.items():
            if data_type in materials_data:
                if data_type == 'material_alternatives':
                    for alt_group in materials_data[data_type]:
                        for alternative in alt_group.get('alternatives', []):
                            documents.append(processor(alternative, alt_group))
                else:
                    for item in materials_data[data_type]:
                        documents.append(processor(item))

        if not documents:
            raise ValueError("No valid documents to process")
        return documents

    @staticmethod
    def _process_product(product: Dict) -> str:
        """
        Formats product information (name, ID, specs, etc.) into a readable text format.
        Used for storing product details in the database.
        """
        return (
            f"Product Information:\n"
            f"Name: {product.get('name', 'N/A')}\n"
            f"ID: {product.get('id', 'N/A')}\n"
            f"Category: {product.get('category', 'N/A')}\n"
            f"Manufacturer: {product.get('manufacturer', 'N/A')}\n"
            f"Specifications: {str(product.get('specifications', {}))}\n"
            f"Applications: {', '.join(product.get('applications', []))}\n"
            f"Technical Details: {str(product.get('technical_details', {}))}\n"
            f"Price History: {str(product.get('price_history', {}))}\n"
            f"Current Stock: {str(product.get('current_stock', {}))}\n"
        )

    @staticmethod
    def _process_technical_doc(doc: Dict) -> str:
        """
        Formats technical documentation into a readable text format.
        Includes document ID, title, and related product information.
        """
        return (
            f"Technical Document:\n"
            f"ID: {doc.get('id', 'N/A')}\n"
            f"Title: {doc.get('title', 'N/A')}\n"
            f"Product ID: {doc.get('product_id', 'N/A')}\n"
            f"Content: {doc.get('content', 'N/A')}\n"
        )

    @staticmethod
    def _process_building_code(code: Dict) -> str:
        """
        Formats building code information into a readable text format.
        Includes code ID, jurisdiction, and which products it applies to.
        """
        return (
            f"Building Code:\n"
            f"Code ID: {code.get('code_id', 'N/A')}\n"
            f"Title: {code.get('title', 'N/A')}\n"
            f"Jurisdiction: {code.get('jurisdiction', 'N/A')}\n"
            f"Applicable Products: {', '.join(code.get('applicable_products', []))}\n"
            f"Summary: {code.get('summary', 'N/A')}\n"
        )

    @staticmethod
    def _process_installation_guide(guide: Dict) -> str:
        """
        Formats installation guides into a readable text format.
        Includes guide ID, title, and related product information.
        """
        return (
            f"Installation Guide:\n"
            f"Guide ID: {guide.get('guide_id', 'N/A')}\n"
            f"Title: {guide.get('title', 'N/A')}\n"
            f"Product ID: {guide.get('product_id', 'N/A')}\n"
            f"Content: {guide.get('content', 'N/A')}\n"
        )

    @staticmethod
    def _process_safety_doc(safety_doc: Dict) -> str:
        """
        Formats safety documentation into a readable text format.
        Includes document ID, title, and related product information.
        """
        return (
            f"Safety Document:\n"
            f"ID: {safety_doc.get('doc_id', 'N/A')}\n"
            f"Title: {safety_doc.get('title', 'N/A')}\n"
            f"Product ID: {safety_doc.get('product_id', 'N/A')}\n"
            f"Content: {safety_doc.get('content', 'N/A')}\n"
        )

    @staticmethod
    def _process_material_alternative(alternative: Dict, alt_group: Dict) -> str:
        """
        Formats information about alternative materials into a readable text format.
        Includes comparison details like durability, cost, and maintenance requirements.
        """
        return (
            f"Material Alternative:\n"
            f"Primary Product ID: {alt_group.get('primary_product_id', 'N/A')}\n"
            f"Alternative Name: {alternative.get('name', 'N/A')}\n"
            f"Alternative ID: {alternative.get('id', 'N/A')}\n"
            f"Comparison:\n"
            f"- Durability: {alternative.get('comparison', {}).get('durability', 'N/A')}\n"
            f"- Cost: {alternative.get('comparison', {}).get('cost', 'N/A')}\n"
            f"- Maintenance: {alternative.get('comparison', {}).get('maintenance', 'N/A')}\n"
            f"- Sustainability: {alternative.get('comparison', {}).get('sustainability', 'N/A')}\n"
            f"- Installation Difficulty: {alternative.get('comparison', {}).get('installation_difficulty', 'N/A')}\n"
        )

    @staticmethod
    def _get_prompt_template() -> str:
        """
        Returns the template used to format questions and context for the AI model.
        This template helps the AI provide consistent and well-structured answers.
        """
        return """Use the following pieces of context and chat history to answer the question at the end. If you find related or overlapping information in the context, use your expertise to make reasonable inferences and calculations to provide a helpful answer.

Chat History:
<chat_history>
{chat_history}
</chat_history>

Context:
<context>
{context}
</context>

Question: 
<question>
{question}
</question>

Instructions:
<instructions>
When answering, follow these guidelines:

1. Response Strategy:
   - If the context contains directly relevant information, provide it first
   - If the context contains related or overlapping information:
     * Use it to make reasonable inferences and calculations
     * Explain your reasoning clearly
     * Cite specific details from the context that informed your answer
   - Only say "I don't know" if there is no relevant or related information in the context

2. Document References:
   Include relevant reference IDs at the beginning of your response for:
   - Product Information: [Product ID: xxx]
   - Technical Documents: [Technical Doc ID: xxx | Product ID: yyy]
   - Building Codes: [Building Code ID: xxx | Applicable Products: yyy,zzz]
   - Installation Guides: [Installation Guide ID: xxx | Product ID: yyy]
   - Safety Documents: [Safety Doc ID: xxx | Product ID: yyy]
   - Material Alternatives: [Alternative Product IDs: xxx,yyy]

3. Answer Structure:
   - Start with the most relevant information
   - Include supporting details from the context
   - Explain any calculations or inferences made
   - Add relevant caveats or considerations
   - Suggest related best practices when applicable

4. Technical Details:
   - Include specific measurements and specifications when available
   - Reference industry standards mentioned in the context
   - Explain technical terms if they appear in the context
   - Connect technical details to practical applications

Remember:
- Base your answer on the context provided, but use your expertise to interpret and apply the information
- Make reasonable inferences when you have related information
- Clearly explain your reasoning when making calculations or recommendations
- If you make an inference, state what context information you based it on
- Only state "I don't know" if there is truly no relevant or related information to work with

</instructions>

Answer:"""

    def create_session(self) -> str:
        """
        Creates a new chat session with a unique ID.
        Keeps track of when the session was created and last used.
        """
        session_id = str(uuid.uuid4())
        self.conversations[session_id] = []
        self.session_metadata[session_id] = {
            'created_at': datetime.now(),
            'last_updated': datetime.now(),
            'message_count': 0
        }
        return session_id

    def get_session(self, session_id: str) -> Optional[List]:
        """
        Retrieves the conversation history for a specific chat session.
        Returns None if the session doesn't exist.
        """
        return self.conversations.get(session_id)

    def add_to_history(self, session_id: str, question: str, answer: str):
        """
        Adds a question and its answer to a chat session's history.
        Also updates the session's last activity timestamp.
        """
        if session_id not in self.conversations:
            raise ValueError("Invalid session ID")
            
        self.conversations[session_id].append({
            'question': question,
            'answer': answer,
            'timestamp': datetime.now()
        })
        
        self.session_metadata[session_id].update({
            'last_updated': datetime.now(),
            'message_count': len(self.conversations[session_id])
        })

    def get_most_recent_session(self) -> str:
        """
        Finds the chat session that was used most recently.
        Creates a new session if none exist.
        """
        if not self.conversations:
            return self.create_session()
        
        return max(
            self.session_metadata.items(),
            key=lambda x: x[1]['last_updated']
        )[0] or self.create_session()

    def query(self, question: str, init_new_session: bool = False) -> dict:
        """
        Processes a question and returns an answer using the stored materials data.
        Can either continue an existing chat session or start a new one.
        """
        try:
            session_id = self.create_session() if init_new_session else self.get_most_recent_session()
            chat_history = [(msg['question'], msg['answer']) for msg in self.conversations[session_id]]
            
            response = self.qa_chain({
                "question": question,
                "chat_history": chat_history
            })
            
            self.add_to_history(session_id, question, response['answer'])
            
            return {
                'session_id': session_id,
                'answer': response['answer'],
                'source_documents': response.get('source_documents', [])
            }

        except Exception as e:
            logger.error(f"Error during query: {e}")
            raise

    async def stream_query(self, question: str, init_new_session: bool = False) -> AsyncGenerator[dict, None]:
        """
        Similar to query(), but streams the answer word by word as it's generated.
        Useful for providing real-time responses in chat interfaces.
        """
        try:
            session_id = self.create_session() if init_new_session else self.get_most_recent_session()
            chat_history = [(msg['question'], msg['answer']) for msg in self.conversations[session_id]]
            
            retrieved_docs = self.qa_chain.retriever.get_relevant_documents(question)
            sources = [str(doc) for doc in retrieved_docs]

            response = await self.qa_chain.acall({
                "question": question, 
                "chat_history": chat_history
            })

            yield {"type": "session_id", "content": session_id}
            yield {"type": "source", "content": sources}

            for word in response['answer'].split():
                yield {"type": "token", "content": word + " "}
                await asyncio.sleep(0.02)

            self.add_to_history(session_id, question, response['answer'])

        except Exception as e:
            logger.error(f"Error during streaming query: {e}")
            yield {"type": "error", "content": str(e)}

    def prepare_documents(self, materials_data: Dict[str, Any]) -> None:
        """
        Prepares and indexes new materials documents into the vector store.
        
        This method processes new materials data and adds it to the existing vector store.
        It handles various types of documents including product catalogs, technical documents,
        building codes, installation guides, safety documents, and material alternatives.
        
        Args:
            materials_data (Dict[str, Any]): Dictionary containing different types of materials data
                Expected format:
                {
                    'product_catalog': List[Dict],
                    'technical_documents': List[Dict],
                    'building_codes': List[Dict],
                    'installation_guides': List[Dict],
                    'safety_documents': List[Dict],
                    'material_alternatives': List[Dict]
                }
        
        Raises:
            ValueError: If materials_data format is invalid or no valid documents are found
            Exception: If document processing or indexing fails
        
        Returns:
            None: Updates the internal vector store with new documents
        """
        try:
            if not isinstance(materials_data, dict):
                raise ValueError("Invalid materials_data format")

            documents = self._prepare_documents(materials_data)
            splits = self.text_splitter.create_documents(documents)
            
            # Add new documents to existing vector store
            self.vector_store.add_documents(documents=splits)
            
            logger.info(f"Successfully prepared and indexed {len(splits)} new documents")
            
        except Exception as e:
            logger.error(f"Error preparing documents: {e}")
            raise