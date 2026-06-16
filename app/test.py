import json
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
import os
from dotenv import load_dotenv
from langchain_nvidia_ai_endpoints import ChatNVIDIA, NVIDIAEmbeddings
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, START
from langgraph.graph.message import add_messages
from typing import TypedDict, Annotated
from pydantic import BaseModel, Field
from langchain_openrouter import ChatOpenRouter
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "..", ".env"))

# ── LLM ──────────────────────────────────────────────────────────────────────
llm = ChatNVIDIA(
    model="qwen/qwen3.5-397b-a17b",
    api_key=os.getenv("NVIDIA_API_KEY"),
    temperature=0,
    top_p=0.95,
    max_completion_tokens=16384,
    # model_kwargs={"enable_thinking": False},
)


# ── Graph ─────────────────────────────────────────────────────────────────────
class State(TypedDict):
    messages: Annotated[list[BaseModel], add_messages]


def chat_node(state: State):
    res = llm.invoke(state["messages"])
    return {"messages": [res]}


graph = (
    StateGraph(State)
    .add_node("chat_node", chat_node)
    .add_edge(START, "chat_node")
    .compile()
)

# # ── Stream test ───────────────────────────────────────────────────────────────
# print("--- streaming start ---")
# for message_chunk, metadata in graph.stream(
#     {"messages": [HumanMessage(content="Write a short note on debugger")]},
#     stream_mode="messages",
# ):
#     if message_chunk.content:
#         print(message_chunk.content, end="|", flush=True)

# print("\n--- streaming end ---")


class OnlyHandyReasearchTopic(BaseModel):
    urls: list[str] = Field(description="List of handy URLs from the data to research")


llm_OnlyHandyReasearchTopic = llm.with_structured_output(OnlyHandyReasearchTopic)


def testingStuff():
    testing_data = [
        {
            "title": "Cockroach Janta Party (CJP): How Abhijeet Dipke's collective ... - BBC",
            "snippet": "The Cockroach Janta Party has used AI-generated images to promote its cause online Indian politics has acquired an unusual mascot: the ...",
            "url": "https://www.bbc.com/news/articles/cz72y11jjq1o",
        },
        {
            "title": "Cockroach Janta Party - Wikipedia",
            "snippet": "The Cockroach Janta Party (CJP; is an Indian satirical political movement founded on 16 May 2026 by Abhijeet Dipke, immediate national outrage",
            "url": "https://en.wikipedia.org/wiki/Cockroach_Janta_Party",
        },
        {
            "title": "Who Are Cockroach Janta Party's 3 Spokespersons? - YouTube",
            "snippet": "CJP Press Conference: Cockroach Janta Party has announced three ... Go to channel DW News · India's 'Cockroach' movement tests its real-world ...",
            "url": "https://www.youtube.com/watch?v=ZP9KuUWmSUA",
        },
        {
            "title": "Cockroach Janta Party: Are You Qualified to Join CJP? - Instagram",
            "snippet": "Today I will say congratulations to you. For people like you, a new political party has arrived in this country named the Cockroach Janta Party.",
            "url": "https://www.instagram.com/reel/DYjr_qElLYM/?hl=en",
        },
        {
            "title": "India's 'Cockroach' CJP party: What investors need to know - CNBC",
            "snippet": "The CJP will face its first offline presence test on Saturday as it plans to hold a protest in New Delhi.",
            "url": "https://www.cnbc.com/2026/06/04/indias-cockroach-cjp-party-what-investors-need-to-know.html",
        },
        {
            "title": "Cockroach Janta Party Holds Protest At Delhi's Jantar Mantar | India ...",
            "snippet": "CJP PROTEST LIVE: The Cockroach Janta Party (CJP) stages protest at Jantar Mantar in New Delhi, demanding the resignation of Union Education ...",
            "url": "https://www.youtube.com/watch?v=0vVfrJJ-Yn0",
        },
        {
            "title": "COCKROACH JANTA PARTY (CJP) : r/delhi - Reddit",
            "snippet": "So yesterday i came across this post on X(twitter) about a new satirical movement or political party named the cockroach janta party.",
            "url": "https://www.reddit.com/r/delhi/comments/1themuu/cockroach_janta_party_cjp/",
        },
        {
            "title": 'Rapid rise of "Cockroach Janta Party" online protest movement ...',
            "snippet": 'Rapid rise of "Cockroach Janta Party" online protest movement appears to spook India\'s leaders. By Arshad R. Zargar. May 22, 2026 / 12:47 PM EDT ...',
            "url": "https://www.cbsnews.com/news/cockroach-janta-party-india-online-protest/",
        },
        {
            "title": "CJP Website Taken Down Amid Crackdown - YouTube",
            "snippet": "The official website of the viral Cockroach Janta Party has been taken offline amid a growing crackdown on the Gen Z-led movement.",
            "url": "https://www.youtube.com/watch?v=ekon9eP7M2o",
        },
    ]

    user_content = f"""
Topic:
CJP (cockorage janta party)

Search Results:
{json.dumps(testing_data, indent=2)}
"""
    response: OnlyHandyReasearchTopic = llm_OnlyHandyReasearchTopic.invoke(
        [
            {
                "role": "system",
                "content": """You are a research URL selection agent.

Your task is to analyze search results and identify the most relevant URLs for the user's topic.
Rules:
- Focus on authoritative, useful, and information-rich sources.
- Ignore low-quality, spammy, or irrelevant websites.
- Return ONLY URLs and maximum only 3 urls.
- Do not explain your choices.
- Do not return titles.
- Do not return snippets.
- Do not use markdown.
- Return one URL per line.""",
            },
            {"role": "user", "content": user_content},
        ]
    )
    print(response, len(response.urls))


testingStuff()