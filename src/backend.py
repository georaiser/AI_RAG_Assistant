"""
Query processing module - simplified version without multi-query expansion.
Handles single query processing with conversational chain.
Backend for the AI RAG Assistant.
"""

from typing import List, Dict, Any, Tuple
from langchain_openai import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain_community.callbacks import get_openai_callback
from vector_store import VectorStoreManager
from config import config
from prompts import get_prompt_template
from pydantic import SecretStr  # Add this import at the top if not present


class QueryProcessor:
    """Handles query processing with retrieval augmentation."""
    
    def __init__(self, vector_store_manager: VectorStoreManager):
        self.vector_store_manager = vector_store_manager
        
        self.llm = ChatOpenAI(
            model=config.MODEL_NAME,
            temperature=config.TEMPERATURE,
            api_key=SecretStr(config.OPENAI_API_KEY)
        )
        self.system_prompt = get_prompt_template("system")
    
    def process_query(self, query: str, chat_history: List[Dict[str, str]]) -> Dict[str, Any]:
        """Process user query and generate response."""
        if not query or not query.strip():
            return self._create_error_response("Please provide a valid question.")
        
        try:
            retriever = self.vector_store_manager.get_retriever()
            if not retriever:
                return self._create_error_response("Search system not available.")
            
            # Create conversational chain
            chain = ConversationalRetrievalChain.from_llm(
                llm=self.llm,
                retriever=retriever,
                combine_docs_chain_kwargs={"prompt": self.system_prompt},
                return_source_documents=True,
                verbose=False
            )
            
            formatted_history = self._format_chat_history(chat_history)
            
            with get_openai_callback() as cb:
                result = chain.invoke({
                    "question": query.strip(),
                    "chat_history": formatted_history
                })
                
                return {
                    "answer": result["answer"],
                    "source_documents": result.get("source_documents", []),
                    "total_tokens": cb.total_tokens,
                    "prompt_tokens": cb.prompt_tokens,
                    "completion_tokens": cb.completion_tokens,
                    "total_cost_usd": cb.total_cost
                }
                
        except Exception as e:
            print(f"Error processing query: {e}")
            return self._create_error_response("I encountered an issue processing your question.")
    
    def _format_chat_history(self, messages: List[Dict[str, str]]) -> List[Tuple[str, str]]:
        """Format chat history for the conversational chain."""
        if not messages or len(messages) < 2:
            return []
        
        formatted_history = []
        # Skip the initial bot message and current user message
        chat_messages = messages[1:-1] if len(messages) > 2 else []
        
        for i in range(0, len(chat_messages), 2):
            if i + 1 < len(chat_messages):
                user_msg = chat_messages[i]
                bot_msg = chat_messages[i + 1]
                
                # Fixed: Check for both 'bot' and 'assistant' roles
                if (user_msg.get('role') == 'user' and 
                    bot_msg.get('role') in ['bot', 'assistant'] and
                    'content' in user_msg and 'content' in bot_msg):
                    formatted_history.append((user_msg['content'], bot_msg['content']))
        
        return formatted_history
    
    def _create_error_response(self, message: str = "") -> Dict[str, Any]:
        """Create standardized error response."""
        error_message = message or f"Hello! I'm {config.BOT_NAME}. I encountered a technical issue. Could you please rephrase your question?"
        
        return {
            "answer": error_message,
            "source_documents": [],
            "total_tokens": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_cost_usd": 0.0
        }