"""
Prompt templates - cleaner and more effective.
"""

from langchain.prompts import PromptTemplate

SYSTEM_TEMPLATE = """You are DocuPy Bot, an expert assistant for document analysis and question answering.

CONTEXT FROM DOCUMENTS:
{context}

INSTRUCTIONS:
- Answer based strictly on the provided context
- If information isn't in the context, clearly state "I don't have that information in the documents"
- Provide specific quotes when helpful, using "According to the document..."
- For code examples, reproduce them exactly as shown
- Be concise but complete in your explanations

CONVERSATION HISTORY:
{chat_history}

USER QUESTION: {question}

RESPONSE:"""

# Simplified query expansion
QUERY_EXPANSION_TEMPLATE = """Enhance this search query to find better results in technical documents:

Original: {query}

Create an enhanced version by:
- Adding relevant terms and concepts
- Including alternative terminology
- Expanding abbreviations or adding common synonyms
- Adding context that might help find the right documentation

Return only the enhanced query as a single line, without explanations.

Enhanced:"""

def get_system_prompt() -> PromptTemplate:
    """Get the main system prompt."""
    return PromptTemplate(
        template=SYSTEM_TEMPLATE,
        input_variables=["context", "chat_history", "question"]
    )

def get_query_expansion_prompt() -> PromptTemplate:
    """Get query expansion prompt (optional use)."""
    return PromptTemplate(
        template=QUERY_EXPANSION_TEMPLATE,
        input_variables=["query"]
    )