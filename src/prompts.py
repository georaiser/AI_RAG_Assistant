"""
Prompt templates module for the query processing system.
Contains all prompt templates used across the application.
"""

from langchain.prompts import PromptTemplate
from config import config


def create_system_prompt() -> PromptTemplate:
    """
    Create the main system prompt template for query processing.
    
    Returns:
        PromptTemplate: Configured system prompt template
    """
    template = f"""You are {config.BOT_NAME}, a helpful Python documentation assistant.

Your role is to answer questions about Python programming using the provided documentation context.

Context from documentation:
{{context}}

Guidelines:
- Use the provided context to answer questions about Python
- If you can find relevant information in the context, provide a clear and helpful answer
- Include code examples from the context when available
- If the specific information is not in the context, say so briefly and provide what related information you can find
- Be conversational and helpful
- Focus on practical, actionable answers

Question: {{question}}

Answer:"""

    return PromptTemplate(
        template=template,
        input_variables=["context", "question"]
    )


# Prompt template registry for easy access
PROMPT_TEMPLATES = {
    "system": create_system_prompt,
}


def get_prompt_template(template_name: str) -> PromptTemplate:
    """
    Get a prompt template by name.
    
    Args:
        template_name: Name of the template to retrieve
        
    Returns:
        PromptTemplate: The requested prompt template
        
    Raises:
        KeyError: If template_name is not found
    """
    if template_name not in PROMPT_TEMPLATES:
        available_templates = ", ".join(PROMPT_TEMPLATES.keys())
        raise KeyError(f"Template '{template_name}' not found. Available templates: {available_templates}")
    
    return PROMPT_TEMPLATES[template_name]()