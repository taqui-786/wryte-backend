import asyncio
import json
import uuid
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.runtime import Runtime
from pydantic import BaseModel
from app.workflow.state import ChatState, UserContext
from app.workflow.tool import (
    MARKDOWN_RULES,
    OnlyHandyReasearchTopic,
    RememberDecision,
    ResearchTopic,
    SummarizedPageContent,
    client,
    llm_OnlyHandyReasearchTopic,
    llm_ResearchTopic,
    llm_SummarizedPageContent,
    llm_classifier_remeber_structured,
    llm_extractor,
    llm_secondary,
    llm_with_tool,
)


# Classifier Node - Just to determine the msg need to be saved Yes / No
class MemoryDecision(BaseModel):
    should_remember: bool
    reason: str


async def classify_node(state: ChatState, runtime: Runtime[UserContext]):
    last_message = state["messages"][-1]
    last_content = (
        last_message.content if hasattr(last_message, "content") else str(last_message)
    )

    if len(last_content) < 5:
        return {"should_remember": False}
    decision: RememberDecision = await llm_classifier_remeber_structured.ainvoke(
        f"Does this message contain personal information about the user worth remembering?\n\n"
        f"YES if the user reveals ANYTHING about themselves - facts, projects, work, "
        f"possessions, preferences, habits, life details, or instructions. "
        f"Even if it's casual or followed by a question (e.g. 'I built my portfolio at X, thoughts?' "
        f"contains 'portfolio at X' → YES).\n"
        f"NO for pure questions, greetings, chitchat, or statements about others.\n\n"
        f'User message: "{last_content}"'
    )
    print(decision.should_remember)
    return {"should_remember": decision.should_remember}


# Recall Node - To just fetch memory for us no reading just giving stuff


async def recall_node(state: ChatState, runtime: Runtime[UserContext]):
    user_id = runtime.context.user_id
    memory_namespace = ("memories", user_id)
    last_message = state["messages"][-1]
    last_content = (
        last_message.content if hasattr(last_message, "content") else str(last_message)
    )
    memories = await runtime.store.asearch(
        memory_namespace, query=last_content, limit=5
    )
    memory_lines = [d.value["data"] for d in memories]
    return {"memories": memory_lines}


SYSTEM_PROMPT_TEMPLATE = """\
You are a helpful, thoughtful writing assistant called Wryte.

You have access to the following long-term memories about the user:
{memory_context}

Use these memories naturally when they are relevant. Do not mention the
memory system explicitly unless asked.

Reply in markdown format. Be concise and to the point. Always think briefly,
then reply with your final content. You have access to tools that can
help you with certain tasks. Use them when needed.

EDITOR TOOLS:
- Use the `read_editor` tool whenever you need to see what the user is
  currently writing before you can help them (summarize, edit, continue,
  review, fix, count words, find typos, etc.). It returns the entire
  current editor content.
- Use the `write_editor` tool to write or replace content in the editor.
  The `content` argument REPLACES the current editor content entirely, so
  if you need to preserve existing text, call `read_editor` first and
  include the existing content in the new value.
- Do not invent or guess editor content. If you are unsure, call
  `read_editor` first.

EDITOR MARKDOWN RULES (the editor only supports the syntax listed below):
{MARKDOWN_RULES}

Always produce content that conforms to these rules. When you write to the
editor, the `content` you pass to `write_editor` MUST be a single
well-formed markdown string using only the inline and block forms above.
"""


# Father Node - Daddy Calling
async def chat_node(state: ChatState, runtime: Runtime[UserContext]):
    memory_lines = state.get("memories", [])
    if memory_lines:
        memory_context = "\n".join(f"- {memory}" for memory in memory_lines)
    else:
        memory_context = "No memories yet."
    print("Messages count:", len(list(state["messages"])))
    system_prompt = SYSTEM_PROMPT_TEMPLATE.format(
        memory_context=memory_context, MARKDOWN_RULES=MARKDOWN_RULES
    )
    response = await llm_with_tool.ainvoke(
        [SystemMessage(content=system_prompt)] + list(state["messages"])
    )
    return {"messages": [response]}


async def remember_node(state: ChatState, runtime: Runtime[UserContext]):
    if not state.get("should_remember"):
        return {}

    # We "schedule" the slow work and return immediately.
    # The user gets their "DONE" signal NOW.
    # The memory saves in the background.
    asyncio.create_task(extract_and_save_node(state, runtime))
    return {}


# Extractor Node - Extract memories from user input
async def extract_and_save_node(state: ChatState, runtime: Runtime[UserContext]):
    if not state.get("should_remember"):
        return {"should_remember": False}

    last_message = state["messages"][-2]
    last_content = (
        last_message.content if hasattr(last_message, "content") else str(last_message)
    )
    memory_namespace = ("memories", runtime.context.user_id)
    extraction_prompt = (
        f'User message: "{last_content}"\n\n'
        "Extract ONLY stable facts about the USER that may be useful in future conversations. "
        "Ignore information about assistants, bots, AI characters, quoted text, greetings, or other people. "
        "Do not guess or infer interests, traits, or preferences. "
        "Return only the extracted facts. If no clear user fact exists, return 'NONE'."
    )
    extraction = await llm_extractor.ainvoke(
        [{"role": "user", "content": extraction_prompt}]
    )
    memory_text = extraction.content.strip()
    if not memory_text:
        return {"should_remember": False}

    if memory_text.upper() == "NONE":
        return {"should_remember": False}

    # Defensive: if the LLM just echoed the user's message back, don't save it
    if memory_text.lower() == last_content.lower():
        return {"should_remember": False}

    if memory_text and memory_text.upper() != "NONE":
        await runtime.store.aput(
            memory_namespace,
            str(uuid.uuid4()),
            {"data": memory_text},
        )
    return {"should_remember": False}


# advance research nodes


async def research_topic_node(state: ChatState, runtime: Runtime[UserContext]) -> dict:
    topic = state["topic"]
    SEARCH_QUERY_SYSTEM_PROMPT = """
You are a search query generation assistant.

Your job is to transform a topic into 2-4 effective web search queries.

Rules:
- Generate 2-3 distinct search queries that cover different angles
- Be specific and include important keywords
- Return as a JSON list of strings: ["query1", "query2", "query3"]
- Do not explain your reasoning.
"""
  
    response = await llm_secondary.ainvoke([  # Use raw LLM
        {"role": "system", "content": SEARCH_QUERY_SYSTEM_PROMPT},
        {"role": "user", "content": topic},
    ])
    
    try:
        queries = json.loads(response.content.strip())
        if not isinstance(queries, list):
            queries = [topic]  # Fallback
    except:
        queries = [topic]  # Fallback
    
    # Ensure 2-4 queries
    queries = queries[:3]
    if len(queries) < 2:
        queries = [topic, f"{topic} overview", f"{topic} latest developments"][:4]
    
    return {"research_topics": queries} 


async def research_node(state: ChatState, runtime: Runtime[UserContext]) -> dict:
    topic = state["topic"]  # this will set by graph - no worries
    try:
        result = client.search.query(query=topic, location="US")
        if not result.results:
            return {"research_results": [f"No search results for:{topic}"]}

        filtered_results = [
            {
                "title": item.title,
                "snippet": item.snippet,
                "url": item.url,
            }
            for item in result.results
        ]

        user_content = f"""
Topic:
{topic}

Search Results:
{json.dumps(filtered_results, indent=2)}
"""
        response: OnlyHandyReasearchTopic = await llm_OnlyHandyReasearchTopic.ainvoke(
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
        urls = response.urls[:3] if response.urls else []
        if not urls:
            return {"research_results": [f"No relevant URLs found for:{topic}"]}

        pages = client.fetch.get_contents(urls=urls, format="markdown")
        valid_pages = []
        for page in pages.results:
            if page.text and len(page.text) > 100:
                valid_pages.append({"url": page.url, "content": page.text})
        if not valid_pages:
            return {"research_results": [f"No relevant content found for:{topic}"]}
        all_summaries = []
        for page in valid_pages:
            summary_user_content = f"""Analyze this page for topic: {topic}
            
URL: {page['url']}
Content: {page['content'][:8000]}  # Truncate per page

"""
            single_topic: SummarizedPageContent = (
                await llm_SummarizedPageContent.ainvoke(
                    [
                        {
                            "role": "system",
                            "content": """
            You are an expert research analyst.

Your task is to analyze webpage content and extract only the information that matters.

Rules:
- Ignore marketing language, sales copy, advertisements, and repetitive content.
- Ignore navigation elements, headers, footers, cookie notices, and unrelated text.
- Focus on factual information, technical details, concepts, workflows, methodologies, features, limitations, and key insights.
- Preserve important statistics, metrics, and examples when present.
- Compress information aggressively while retaining meaning.
- Do not copy large portions of the original text.
- Produce concise but information-dense summaries.

For each webpage return:

1. Source URL
2. One-sentence summary
3. Key insights (bullet list)
4. Important technical details
5. Important features or capabilities
6. Notable limitations, risks, or constraints
7. Final condensed research summary

Your output should prioritize information quality over length.
            """,
                        },
                        {"role": "user", "content": summary_user_content},
                    ]
                )
            )
            all_summaries.append(single_topic.summary)
        return {"research_results": all_summaries}
    except Exception as e:
        print(f"Error in research_node for '{topic}': {e}")
        return {"research_results": [f"Research failed for {topic}: {str(e)}"]}

async def finalize_research(state: ChatState) -> dict:
    all_summaries = state.get("research_results", [])
    
    if not all_summaries:
        return {"final_research_report": "No research results to synthesize."}
    
    # Combine all summaries into a structured report
    combined_content = "\n\n---\n\n".join(all_summaries)
    
    # Use LLM to create final polished report
    synthesis_prompt = f"""Create a comprehensive research report from these summaries:

{combined_content}

Structure the report as:
1. Executive Summary (2-3 sentences)
2. Key Findings (bulleted)
3. Detailed Analysis (organized by subtopic)
4. Important Statistics/Data Points
5. Limitations/Gaps
6. Sources Referenced

Be concise but thorough. Use markdown formatting."""
    
    try:
        
        response = await llm_secondary.ainvoke([
            HumanMessage(content=synthesis_prompt)
        ])
        
        return {"final_research_report": response.content}
    except Exception as e:
        print(f"Error in finalize_research: {e}")
        # Fallback: simple concatenation
        return {"final_research_report": f"Research Report:\n\n{combined_content}"}


async def research_answer_node(state:ChatState):
    report = state['final_research_report']
    print("REPORT ----->>>>", report)
    response = await llm_secondary.ainvoke([
        SystemMessage(content="Answer the user using the complete research. Do not call Tools."),
        HumanMessage(content=report)
    ])
    return {"messages": [response]}