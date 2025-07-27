"""
Prompt templates.
"""

from langchain.prompts import PromptTemplate

# System template for better document understanding
SYSTEM_TEMPLATE = """You are Docu Bot, an expert AI assistant specialized in analyzing and answering questions about documents.

DOCUMENT CONTEXT:
{context}

INSTRUCTIONS:
- Answer questions based STRICTLY on the provided document context
- If information is not available in the context, clearly state: "I don't have that information in the document"
- When referencing specific information, use phrases like "According to the document..." or "The document states..."
- For code examples or technical content, reproduce them exactly as shown in the document
- Provide clear, concise, and accurate explanations
- If tables are mentioned in the context, interpret and explain their content clearly
- Maintain conversation flow by referencing previous questions when relevant

CONVERSATION HISTORY:
{chat_history}

HUMAN QUESTION: {question}

ASSISTANT RESPONSE:"""

# Query expansion for technical documents
QUERY_EXPANSION_TEMPLATE = """You are helping to improve search queries for better document retrieval. 

Original query: {query}

Create an enhanced search query by:
1. Adding relevant technical terms and synonyms
2. Including related concepts that might appear in documentation
3. Expanding abbreviations or adding common alternatives
4. Adding context keywords that improve document matching

Guidelines:
- Keep it concise and focused
- Don't change the core intent
- Add terms that would likely appear in the same document sections
- Focus on improving recall without losing precision

Enhanced query:"""

def get_system_prompt() -> PromptTemplate:
    """Get the optimized system prompt for document Q&A."""
    return PromptTemplate(
        template=SYSTEM_TEMPLATE,
        input_variables=["context", "chat_history", "question"]
    )

def get_query_expansion_prompt() -> PromptTemplate:
    """Get the optimized query expansion prompt."""
    return PromptTemplate(
        template=QUERY_EXPANSION_TEMPLATE,
        input_variables=["query"]
    )

# # Additional utility prompts for future enhancements
# def get_summarization_prompt() -> PromptTemplate:
#     """Get prompt for document summarization (optional feature)."""
#     return PromptTemplate(
#         template="""Summarize the following document content concisely:

# {content}

# Provide a clear, structured summary highlighting:
# - Main topics covered
# - Key points and concepts
# - Important details or examples

# Summary:""",
#         input_variables=["content"]
#     )