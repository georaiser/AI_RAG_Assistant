# backend.py
"""
Simplified RAG Engine with better error handling.
"""

import logging
from typing import Dict, List, Any
from langchain_openai import ChatOpenAI
from langchain_community.callbacks import get_openai_callback
from langchain.prompts import PromptTemplate
from langchain.schema import HumanMessage
from vector_store import VectorStoreManager
from prompts import REWRITE_QUESTION_PROMPT, SYSTEM_TEMPLATE, SUMMARY_TEMPLATE
from config import Config
from pydantic import SecretStr

logger = logging.getLogger(__name__)


class RAGEngine:
    """Simplified RAG engine for document question-answering."""
    
    def __init__(self, vector_manager: VectorStoreManager):
        """Initialize RAG engine."""
        self.vector_manager = vector_manager
        self.llm = ChatOpenAI(
            model=Config.MODEL_NAME,
            temperature=Config.TEMPERATURE,
            api_key=SecretStr(Config.OPENAI_API_KEY)
        )
        logger.info(f"RAG Engine initialized with {Config.MODEL_NAME}")
    
    def process_query(self, query: str, chat_history: List[Dict]) -> Dict[str, Any]:
        """Process user query and return response."""
        try:
            # Check if vector store is ready
            if not self.vector_manager.is_ready():
                return self._error_response("Vector store not ready")
            
            with get_openai_callback() as cb:
                # Rewrite query for conversational context
                final_query = self._rewrite_query(query, chat_history)

                # Get retriever and search documents
                retriever = self.vector_manager.get_retriever()
                if not retriever:
                    return self._error_response("Could not create retriever")
                
                docs = retriever.invoke(final_query)
                
                if not docs:
                    return {
                        "answer": "No relevant information found. Try rephrasing your question.",
                        "total_tokens": 0,
                        "total_cost_usd": 0.0
                    }
                
                # Create context
                context = self._create_context(docs)
                history = self._format_history(chat_history)
                
                # Log conversation context for debugging
                logger.info(f"Processing query: {query[:100]}...")
                logger.info(f"History length: {len(history)} characters")
                logger.info(f"Context length: {len(context)} characters")
                
                # Log conversation context for natural flow
                if history and history != "No previous conversation.":
                    logger.info("Conversation history available - maintaining context")
                
                # Log context quality for citations
                if docs:
                    logger.info(f"Retrieved {len(docs)} documents for context")
                    # Check if documents have page information
                    docs_with_pages = sum(1 for doc in docs if doc.metadata.get('page'))
                    logger.info(f"Documents with page info: {docs_with_pages}/{len(docs)}")
                else:
                    logger.warning("No documents retrieved - may affect citation quality")
                

                
                # Generate response
                prompt = PromptTemplate(
                    template=SYSTEM_TEMPLATE,
                    input_variables=["context", "chat_history", "question"]
                )
                
                prompt_text = prompt.format(
                    context=context,
                    chat_history=history,
                    question=query
                )
                
                response = self.llm.invoke([HumanMessage(content=prompt_text)])
                
                return {
                    "answer": response.content,
                    "total_tokens": cb.total_tokens,
                    "prompt_tokens": cb.prompt_tokens,
                    "completion_tokens": cb.completion_tokens,
                    "total_cost_usd": cb.total_cost
                }
                
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            return self._error_response(f"Processing error: {str(e)}")
    
    def generate_summary(self) -> Dict[str, Any]:
        """Generate document summary."""
        try:
            if not self.vector_manager.is_ready():
                return self._error_response("Vector store not ready")
            
            with get_openai_callback() as cb:
                retriever = self.vector_manager.get_retriever()
                if not retriever:
                    return self._error_response("Could not create retriever")
                
                # Get diverse documents for summary
                docs = retriever.invoke("overview methods procedures implementation results")
                
                if not docs:
                    return self._error_response("No documents for summary")
                
                context = self._create_context(docs)
                
                prompt = PromptTemplate(template=SUMMARY_TEMPLATE, input_variables=["context"])
                prompt_text = prompt.format(context=context)
                
                response = self.llm.invoke([HumanMessage(content=prompt_text)])
                
                return {
                    "answer": response.content,
                    "total_tokens": cb.total_tokens,
                    "prompt_tokens": cb.prompt_tokens,
                    "completion_tokens": cb.completion_tokens,
                    "total_cost_usd": cb.total_cost
                }
                
        except Exception as e:
            logger.error(f"Error generating summary: {e}")
            return self._error_response(f"Summary error: {str(e)}")
    
    def _create_context(self, documents: List) -> str:
        """Create context from retrieved documents, ensuring unique headers."""
        if not documents:
            return "No documents found."

        context_parts: List[str] = []
        seen_headers: set[str] = set()

        for doc in documents:
            metadata = doc.metadata or {}
            source = metadata.get("source", "Unknown")
            # Some loaders store page information under different keys
            page = metadata.get("page") or metadata.get("page_number")
            content_type = metadata.get("content_type", "text")

            # Build header
            if page is not None:
                header = f"[{content_type.upper()} - Source: {source}, Page: {page}]"
            else:
                header = f"[{content_type.upper()} - Source: {source}]"

            # Add to context only if header not yet used (prevents duplication)
            if header not in seen_headers:
                context_parts.append(f"{header}\n{doc.page_content}\n")
                seen_headers.add(header)

        return "\n".join(context_parts)

    def _rewrite_query(self, query: str, chat_history: List[Dict]) -> str:
        """Rewrite a follow-up query to be a standalone query."""
        # If history is short, no need to rewrite
        if len(chat_history) <= 1:
            return query

        history = self._format_history(chat_history, for_rewrite=True)
        if history == "No previous conversation.":
            return query

        prompt = PromptTemplate(
            template=REWRITE_QUESTION_PROMPT,
            input_variables=["chat_history", "question"],
        )
        prompt_text = prompt.format(chat_history=history, question=query)

        try:
            logger.info(f"Rewriting query: '{query}'")
            response = self.llm.invoke([HumanMessage(content=prompt_text)])
            rewritten_query = response.content.strip()

            # Basic validation: if the model returns an empty string or the original query, stick with the original
            if rewritten_query and rewritten_query.lower() != query.lower():
                logger.info(f"Rewritten query: '{rewritten_query}'")
                return rewritten_query
            else:
                logger.info("Query rewrite returned original or empty, using original.")
                return query
        except Exception as e:
            logger.error(f"Error during query rewrite: {e}")
            return query  # Fallback to original query on error
    
    def _format_history(self, messages: List[Dict], for_rewrite: bool = False) -> str:
        """Format chat history with proper role handling."""
        if len(messages) <= 1:
            return "No previous conversation."
        
        # For query rewriting, we include the last user message
        end_offset = 0 if for_rewrite else 1

        # Get relevant messages (skip welcome message and current message)
        relevant = []
        for i, msg in enumerate(messages):
            # Skip system/welcome message and current user message (if not for_rewrite)
            if i == 0 or i == len(messages) - end_offset:
                continue
            if not self._is_welcome_message(msg):
                relevant.append(msg)

        # Take last 3 exchanges (6 messages: 3 user + 3 assistant) to understand conversation flow
        if len(relevant) > 6:
            relevant = relevant[-6:]

        history_parts = []
        i = 0
        while i < len(relevant):
            # Find next user and assistant messages in order
            if relevant[i]["role"] == "user":
                user_msg = relevant[i]
                i += 1
                # Look ahead for next assistant message
                assistant_msg = None
                while i < len(relevant) and relevant[i]["role"] != "assistant":
                    i += 1
                if i < len(relevant) and relevant[i]["role"] == "assistant":
                    assistant_msg = relevant[i]
                    i += 1
                else:
                    assistant_msg = None
                # Add the exchange
                user_content = (
                    user_msg['content'][:200] + "..."
                    if len(user_msg['content']) > 200
                    else user_msg['content']
                )
                if assistant_msg:
                    if len(assistant_msg['content']) > 400:
                        head = assistant_msg['content'][:250]
                        tail = assistant_msg['content'][-120:]
                        assistant_content = f"{head}…{tail}"
                    else:
                        assistant_content = assistant_msg['content']
                    history_parts.append(f"Human: {user_content}")
                    history_parts.append(f"Assistant: {assistant_content}")
                else:
                    history_parts.append(f"Human: {user_content}")
            else:
                # If role is assistant and not paired, just add as assistant
                assistant_msg = relevant[i]
                i += 1
                if len(assistant_msg['content']) > 400:
                    head = assistant_msg['content'][:250]
                    tail = assistant_msg['content'][-120:]
                    assistant_content = f"{head}…{tail}"
                else:
                    assistant_content = assistant_msg['content']
                history_parts.append(f"Assistant: {assistant_content}")

        return "\n".join(history_parts) if history_parts else "No previous conversation."

    
    def _is_welcome_message(self, message: Dict) -> bool:
        """Check if message is welcome message."""
        content = message.get('content', '').lower()
        return 'welcome' in content or 'technical document assistant' in content
    

    
    def _error_response(self, message: str) -> Dict[str, Any]:
        """Create error response."""
        return {
            "answer": f"Error: {message}. Please try again.",
            "total_tokens": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_cost_usd": 0.0
        }


def main():
    """Main function."""
    app = TechnicalDocumentApp()
    app.run()


if __name__ == "__main__":
    main()