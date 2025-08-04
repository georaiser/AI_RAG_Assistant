# langgraph_backend.py
"""
Simple LangGraph RAG implementation replacing the original backend.py
"""

import logging
from typing import Dict, List, Any, TypedDict, Annotated
from langchain_openai import ChatOpenAI
from langchain_community.callbacks import get_openai_callback
from langchain.schema import HumanMessage, AIMessage
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from vector_store import VectorStoreManager
from prompts import SYSTEM_TEMPLATE, SUMMARY_TEMPLATE
from config import Config
from pydantic import SecretStr

logger = logging.getLogger(__name__)

# Simple role constants
USER_ROLE = "user"
ASSISTANT_ROLE = "assistant"


class GraphState(TypedDict):
    """State for the RAG graph."""
    messages: Annotated[List, add_messages]
    query: str
    context: str
    answer: str
    source_documents: List
    total_tokens: int
    total_cost_usd: float


class LangGraphRAGEngine:
    """Simple LangGraph RAG engine."""
    
    def __init__(self, vector_manager: VectorStoreManager):
        """Initialize the RAG engine."""
        self.vector_manager = vector_manager
        self.llm = ChatOpenAI(
            model=Config.MODEL_NAME,
            temperature=Config.TEMPERATURE,
            api_key=SecretStr(Config.OPENAI_API_KEY)
        )
        self._retriever = None
        self.graph = self._build_graph()
        logger.info(f"LangGraph RAG Engine initialized with {Config.MODEL_NAME}")
    
    def _get_retriever(self):
        """Get or create retriever (cached)."""
        if self._retriever is None:
            self._retriever = self.vector_manager.get_retriever()
        return self._retriever
    
    def _build_graph(self) -> StateGraph:
        """Build the RAG graph."""
        # Create graph
        workflow = StateGraph(GraphState)
        
        # Add nodes
        workflow.add_node("retrieve", self._retrieve_documents)
        workflow.add_node("generate", self._generate_answer)
        
        # Define the flow
        workflow.set_entry_point("retrieve")
        workflow.add_edge("retrieve", "generate")
        workflow.add_edge("generate", END)
        
        return workflow.compile()
    
    def _retrieve_documents(self, state: GraphState) -> GraphState:
        """Retrieve relevant documents."""
        try:
            retriever = self._get_retriever()
            if not retriever:
                state["source_documents"] = []
                state["context"] = "No documents found."
                return state
            
            # Retrieve documents
            docs = retriever.invoke(state["query"])
            state["source_documents"] = docs
            
            # Create context
            state["context"] = self._create_context(docs)
            
            if Config.DEBUG_MODE:
                logger.info(f"Retrieved {len(docs)} documents")
            
            return state
            
        except Exception as e:
            logger.error(f"Error in retrieve step: {e}")
            state["source_documents"] = []
            state["context"] = "Error retrieving documents."
            return state
    
    def _generate_answer(self, state: GraphState) -> GraphState:
        """Generate answer using LLM."""
        try:
            # Format chat history
            chat_history = self._format_chat_history(state["messages"])
            
            # Create prompt
            prompt_text = SYSTEM_TEMPLATE.format(
                context=state["context"],
                chat_history=chat_history,
                question=state["query"]
            )
            
            # Generate response with callback
            with get_openai_callback() as cb:
                response = self.llm.invoke([HumanMessage(content=prompt_text)])
                
                state["answer"] = response.content
                state["total_tokens"] = cb.total_tokens
                state["total_cost_usd"] = cb.total_cost
            
            if Config.DEBUG_MODE:
                logger.info(f"Generated answer: {len(state['answer'])} chars")
            
            return state
            
        except Exception as e:
            logger.error(f"Error in generate step: {e}")
            state["answer"] = f"I apologize, but I encountered an error: {str(e)}"
            state["total_tokens"] = 0
            state["total_cost_usd"] = 0.0
            return state
    
    def _create_context(self, documents: List) -> str:
        """Create context from documents with proper citations."""
        if not documents:
            return "No documents found."
        
        # Group documents by source file
        docs_by_source = {}
        for doc in documents:
            source = doc.metadata.get('source', 'Unknown')
            if source not in docs_by_source:
                docs_by_source[source] = []
            docs_by_source[source].append(doc)
        
        context_parts = []
        total_chars = 0
        max_chars = Config.MAX_CONTEXT_CHARS
        
        # Process each source file
        for source, docs in docs_by_source.items():
            if total_chars >= max_chars:
                break
                
            # Sort documents by page number if available
            docs.sort(key=lambda d: d.metadata.get('page', 0))
            
            # Process each document
            for doc in docs:
                if total_chars >= max_chars:
                    break
                
                content = doc.page_content.strip()
                if not content:
                    continue
                
                # Ensure proper citation header
                content = self._ensure_citation_header(content, doc.metadata)
                
                # Check if adding this would exceed limit
                if total_chars + len(content) + 2 > max_chars:
                    break
                
                context_parts.append(content)
                total_chars += len(content) + 2
        
        final_context = "\n\n".join(context_parts)
        
        if Config.DEBUG_MODE:
            logger.info(f"Created context with {len(context_parts)} parts, {total_chars} chars")
        
        return final_context
    
    def _ensure_citation_header(self, content: str, metadata: Dict) -> str:
        """Ensure content has proper citation header."""
        lines = content.split('\n')
        first_line = lines[0] if lines else ""
        
        # Get source info
        source = metadata.get('source', 'Unknown')
        page = metadata.get('page')
        
        # Create proper header
        if page is not None:
            correct_header = f"[{source}, Page: {page}]"
        else:
            correct_header = f"[{source}]"
        
        # Check if first line is already a proper citation
        if first_line.startswith('[') and ']' in first_line and source in first_line:
            # Update existing header to ensure page is included if available
            if page is not None and f"Page: {page}" not in first_line:
                content_body = '\n'.join(lines[1:]).strip()
                return f"{correct_header}\n{content_body}"
            return content
        else:
            # Remove any existing bad header and add correct one
            if first_line.startswith('[') and ']' in first_line:
                content_body = '\n'.join(lines[1:]).strip()
            else:
                content_body = content.strip()
            
            return f"{correct_header}\n{content_body}"
    
    def _format_chat_history(self, messages: List) -> str:
        """Format chat history for the prompt."""
        if len(messages) <= 1:
            return ""
        
        # Filter messages - exclude welcome message and current query
        filtered_messages = []
        for i, msg in enumerate(messages):
            # Skip welcome message (first message with "Welcome")
            if i == 0 and "Welcome" in msg.content:
                continue
            # Skip current query (last message)
            if i == len(messages) - 1:
                continue
            filtered_messages.append(msg)
        
        if not filtered_messages:
            return ""
        
        # Keep only last 6 messages (3 exchanges) to prevent context overflow
        if len(filtered_messages) > 6:
            filtered_messages = filtered_messages[-6:]
        
        # Format as conversation
        formatted_parts = []
        for msg in filtered_messages:
            role = "Human" if msg.type == "human" else "Assistant"
            content = msg.content[:400]  # Truncate long messages
            formatted_parts.append(f"{role}: {content}")
        
        return "\n".join(formatted_parts)
    
    def process_query(self, query: str, chat_history: List[Dict]) -> Dict[str, Any]:
        """Process a user query using the graph."""
        try:
            # Convert chat history to LangChain messages
            messages = []
            for msg in chat_history[:-1]:  # Exclude current query
                if msg["role"] == USER_ROLE:
                    messages.append(HumanMessage(content=msg["content"]))
                elif msg["role"] == ASSISTANT_ROLE:
                    messages.append(AIMessage(content=msg["content"]))
            
            # Initial state
            initial_state = {
                "messages": messages,
                "query": query,
                "context": "",
                "answer": "",
                "source_documents": [],
                "total_tokens": 0,
                "total_cost_usd": 0.0
            }
            
            # Run the graph
            final_state = self.graph.invoke(initial_state)
            
            return {
                "answer": final_state["answer"],
                "source_documents": final_state["source_documents"],
                "total_tokens": final_state["total_tokens"],
                "prompt_tokens": 0,  # Not available in this simple implementation
                "completion_tokens": 0,  # Not available in this simple implementation
                "total_cost_usd": final_state["total_cost_usd"]
            }
            
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            return self._error_response(f"Processing error: {str(e)}")
    
    def generate_summary(self) -> Dict[str, Any]:
        """Generate document summary using the graph."""
        try:
            if not self.vector_manager.is_ready():
                return self._error_response("Vector store not ready")
            
            # Create a simple summary graph
            summary_workflow = StateGraph(GraphState)
            summary_workflow.add_node("retrieve_for_summary", self._retrieve_for_summary)
            summary_workflow.add_node("generate_summary", self._generate_summary_response)
            
            summary_workflow.set_entry_point("retrieve_for_summary")
            summary_workflow.add_edge("retrieve_for_summary", "generate_summary")
            summary_workflow.add_edge("generate_summary", END)
            
            summary_graph = summary_workflow.compile()
            
            # Run summary generation
            initial_state = {
                "messages": [],
                "query": "overview summary content",
                "context": "",
                "answer": "",
                "source_documents": [],
                "total_tokens": 0,
                "total_cost_usd": 0.0
            }
            
            final_state = summary_graph.invoke(initial_state)
            
            return {
                "answer": final_state["answer"],
                "total_tokens": final_state["total_tokens"],
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_cost_usd": final_state["total_cost_usd"]
            }
            
        except Exception as e:
            logger.error(f"Error generating summary: {e}")
            return self._error_response(f"Summary error: {str(e)}")
    
    def _retrieve_for_summary(self, state: GraphState) -> GraphState:
        """Retrieve documents for summary."""
        try:
            retriever = self._get_retriever()
            if not retriever:
                state["source_documents"] = []
                state["context"] = "No documents found."
                return state
            
            # Retrieve more documents for summary
            docs = retriever.invoke(state["query"])
            
            if len(docs) > Config.SUMMARY_K:
                docs = docs[:Config.SUMMARY_K]
            
            state["source_documents"] = docs
            state["context"] = self._create_context(docs)
            
            if Config.DEBUG_MODE:
                logger.info(f"Retrieved {len(docs)} documents for summary")
            
            return state
            
        except Exception as e:
            logger.error(f"Error retrieving for summary: {e}")
            state["source_documents"] = []
            state["context"] = "Error retrieving documents."
            return state
    
    def _generate_summary_response(self, state: GraphState) -> GraphState:
        """Generate summary response."""
        try:
            prompt_text = SUMMARY_TEMPLATE.format(context=state["context"])
            
            with get_openai_callback() as cb:
                response = self.llm.invoke([HumanMessage(content=prompt_text)])
                
                state["answer"] = response.content
                state["total_tokens"] = cb.total_tokens
                state["total_cost_usd"] = cb.total_cost
            
            if Config.DEBUG_MODE:
                logger.info(f"Generated summary: {len(state['answer'])} chars")
            
            return state
            
        except Exception as e:
            logger.error(f"Error generating summary: {e}")
            state["answer"] = f"I apologize, but I encountered an error generating the summary: {str(e)}"
            state["total_tokens"] = 0
            state["total_cost_usd"] = 0.0
            return state
    
    def _error_response(self, message: str) -> Dict[str, Any]:
        """Create standardized error response."""
        error_msg = f"I apologize, but I encountered an error: {message}. Please try rephrasing your question."
        
        if Config.DEBUG_MODE:
            logger.error(f"Error response: {message}")
        
        return {
            "answer": error_msg,
            "source_documents": [],
            "total_tokens": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_cost_usd": 0.0
        }


# Alias for backward compatibility
RAGEngine = LangGraphRAGEngine