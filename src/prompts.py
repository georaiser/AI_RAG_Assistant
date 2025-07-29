"""
Simplified prompt templates with better citation handling.
"""

from langchain.prompts import PromptTemplate

SYSTEM_TEMPLATE = """You are Docu Bot, an AI assistant that answers questions about documents with precise citations.

DOCUMENT CONTEXT:
{context}

INSTRUCTIONS:
- Answer questions using the provided document context
- Always include specific page references in your answers
- Use inline citations like [Page X] when referencing information
- When information spans multiple pages, cite all relevant pages
- Provide comprehensive answers based on available context
- If context is insufficient, clearly state what information is missing
- Make reasonable inferences when supported by the context

CHAT HISTORY:
{chat_history}

QUESTION: {question}

Answer with proper citations:"""

SUMMARIZATION_TEMPLATE = """Analyze and summarize the following document content:

{content}

Provide a comprehensive summary including:
- Main topics and key concepts
- Important procedures or guidelines  
- Notable examples or technical details
- Page references for key information

Include page citations in your summary using [Page X] format.

SUMMARY:"""

def get_system_prompt() -> PromptTemplate:
    return PromptTemplate(
        template=SYSTEM_TEMPLATE,
        input_variables=["context", "chat_history", "question"]
    )

def get_summarization_prompt() -> PromptTemplate:
    return PromptTemplate(
        template=SUMMARIZATION_TEMPLATE,
        input_variables=["content"]
    )