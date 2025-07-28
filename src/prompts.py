"""
Prompt templates for better document understanding and retrieval.
"""

from langchain.prompts import PromptTemplate

SYSTEM_TEMPLATE = """You are Docu Bot, an expert AI assistant specialized in analyzing and answering questions about documents. You are helpful, knowledgeable, and provide comprehensive answers based on the available context.

DOCUMENT CONTEXT:
{context}

INSTRUCTIONS:
- Answer questions using the provided document context as your primary source
- When the context contains relevant information, provide detailed and comprehensive answers
- If the exact answer isn't in the context but related information is available, use that information to provide a helpful response and mention what specific information might be missing
- You can make reasonable inferences and connections based on the context provided
- When referencing information, you can say "Based on the document..." or "According to the information provided..."
- For code examples, technical procedures, or specific details, reproduce them accurately from the context
- If tables or structured data are mentioned, interpret and explain them clearly
- Provide context and background information when it helps explain the answer
- Be conversational and maintain the flow of discussion by referencing previous questions when relevant
- Only say you don't have information if the context is completely unrelated to the question
- If you can provide a partial answer or related information, do so and explain what additional details might be needed

CONVERSATION HISTORY:
{chat_history}

HUMAN QUESTION: {question}

Provide a helpful and comprehensive response based on the available context:"""

# Enhanced query expansion for better retrieval
QUERY_EXPANSION_TEMPLATE = """You are helping to improve search queries for better document retrieval. Your goal is to find the most relevant information in the document.

Original query: {query}

Create 2-3 enhanced search variations by:
1. Adding synonyms and related technical terms
2. Breaking complex questions into key concepts
3. Including alternative phrasings and common terminology
4. Adding context words that might appear in relevant sections
5. Including both specific and general terms

Guidelines:
- Focus on terms likely to appear in the document
- Include both technical and plain language versions
- Consider different ways the information might be presented
- Keep each variation focused and searchable

Return the variations as separate lines, with the most promising one first:"""

# Fallback query template for when initial retrieval fails
FALLBACK_QUERY_TEMPLATE = """The user asked: "{original_query}"

The initial search didn't find good matches. Generate 3 broader search terms that might capture related information:
1. A very general version focusing on the main topic
2. A version with common synonyms and alternative terms  
3. A version focusing on key concepts and categories

Format as simple search terms, one per line:"""

def get_system_prompt() -> PromptTemplate:
    """Get the improved system prompt for document Q&A."""
    return PromptTemplate(
        template=SYSTEM_TEMPLATE,
        input_variables=["context", "chat_history", "question"]
    )

def get_query_expansion_prompt() -> PromptTemplate:
    """Get the improved query expansion prompt."""
    return PromptTemplate(
        template=QUERY_EXPANSION_TEMPLATE,
        input_variables=["query"]
    )

def get_fallback_query_prompt() -> PromptTemplate:
    """Get prompt for fallback query generation."""
    return PromptTemplate(
        template=FALLBACK_QUERY_TEMPLATE,
        input_variables=["original_query"]
    )

def get_summarization_prompt() -> PromptTemplate:
    """Get prompt for document summarization."""
    return PromptTemplate(
        template="""Analyze and summarize the following document content:

{content}

Provide a comprehensive summary that includes:

**Main Topics & Structure:**
- Key sections and their focus areas
- Primary themes and concepts covered

**Important Information:**
- Critical facts, procedures, or guidelines
- Key technical details or specifications
- Important examples or use cases

**Notable Features:**
- Special sections, tables, or formatted content
- Code examples, formulas, or technical diagrams
- Reference materials or additional resources mentioned

Keep the summary informative and well-organized, highlighting the most valuable content for someone trying to understand what the document contains.

SUMMARY:""",
        input_variables=["content"]
    )

# Context enhancement prompt for better understanding
def get_context_enhancement_prompt() -> PromptTemplate:
    """Get prompt for enhancing context understanding."""
    return PromptTemplate(
        template="""Given this document context and user question, identify the key information and relationships:

CONTEXT: {context}
QUESTION: {question}

Extract and organize:
1. **Direct answers:** Information that directly answers the question
2. **Related information:** Context that supports or explains the answer
3. **Key concepts:** Important terms or ideas mentioned
4. **Connections:** How different pieces of information relate to each other

Provide a structured analysis that will help generate a comprehensive response:""",
        input_variables=["context", "question"]
    )