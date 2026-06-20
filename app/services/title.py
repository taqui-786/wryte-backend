from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from app.workflow.llm import llm_secondary


async def generate_title_for_chat(conversation: str) -> str:
    prompt = ChatPromptTemplate.from_template(
        """
You are an AI assistant that generates concise chat titles for a writing editor application.

Your task is to create a short, natural, and meaningful title based on:
1. The user's first message
2. The assistant's initial response

The title should summarize the main intent or topic of the conversation, similar to how ChatGPT names chats.

Rules:
- Maximum 5 words
- Clear and human-friendly
- Do not use quotes
- Avoid generic titles like "New Chat" or "Conversation"
- Avoid unnecessary filler words
- Return only the title

Conversation:
{conversation}

Title:
"""
    )
    output_parser = StrOutputParser()
    chain = prompt | llm_secondary | output_parser
    response = await chain.ainvoke({"conversation": conversation})
    return response



class TitleService:
    """Orchestrates title operations"""
    
    @staticmethod
    async def generate_title(content: str) -> str:
        return await generate_title_for_chat(content)
        