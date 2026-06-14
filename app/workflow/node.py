import asyncio
import json
import uuid
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.runtime import Runtime
from pydantic import BaseModel
from app.workflow.state import ChatState, UserContext
from app.workflow.tool import (
    MARKDOWN_RULES,
    OnlyHandyReasearchTopic,
    RememberDecision,
    StyleCheck,
    SummarizedPageContent,
    WritingPlan,
    client,
    llm_OnlyHandyReasearchTopic,
    llm_SummarizedPageContent,
    llm_WriterPlan,
    llm_classifier,
    llm_classifier_remeber_structured,
    llm_extractor,
    llm_secondary,
    llm_style_check,
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
        f"You are a memory classifier for a personal AI assistant.\n\n"
        f"REMEMBER (True) if the user reveals ANYTHING about themselves:\n"
        f"- Facts, projects, work, skills, habits, preferences, opinions, possessions\n"
        f"- Instructions for how the AI should behave\n"
        f"- Even casual mentions mid-question ('I built my portfolio in Next.js, thoughts?' → True)\n"
        f"- Long or detailed messages that reveal writing style, tone, or vocabulary\n\n"
        f"SKIP (False) only for:\n"
        f"- Pure questions with zero self-disclosure ('What is RAG?')\n"
        f"- Greetings, filler, thanks ('hey', 'ok', 'thanks')\n"
        f"- Statements purely about others or external topics\n\n"
        f"STYLE NOTE: If the message is 30+ words or has a distinctive voice/tone, "
        f"mention it in the reason field — e.g. 'casual technical tone, thinks out loud, worth style profiling.'\n\n"
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


EDITOR_CHANGE_GUIDELINES = """\
EDITOR CHANGE RULES:
When editing EXISTING content, use `update_editor` with ONLY the changed lines.

Workflow:
1. Call `read_editor` first to see current content.
2. Identify ONLY the lines that need to change.
3. Call `update_editor(changes=[...])` with just those changes.
   DO NOT return the full document — only what changed.

Rules:
- Line = one block (paragraph, heading, code block, list, etc.).
  Blocks are separated by blank lines in markdown.
- Line numbers are 1-indexed.
- Types: "replace" (swap content at line), "delete" (remove line),
  "insert" (add new content AFTER that line number).
- If asked to do a small edit on existing content, NEVER rewrite
  the whole thing. Use update_editor with 1-3 minimal changes.

Examples:
- "Make intro shorter" → changes=[{"line": 2, "type": "replace",
  "content": "Short intro."}]
- "Add a conclusion" → changes=[{"line": 8, "type": "insert",
  "content": "## Conclusion\\n\\nConcluding paragraph."}]
- "Remove 3rd para" → changes=[{"line": 3, "type": "delete", "content": ""}]
"""

SYSTEM_PROMPT_TEMPLATE = """\
You are Wryte — a writing assistant built into a markdown editor.
You do NOT chat generically. You help the user write, edit, refine,
and research content inside their editor. Everything the user refers
to is about what's in that editor. If you are confuse just stop and Ask user to be more specific.

The user's content lives in the editor. That is your workspace.

YOUR IDENTITY:
- You are an extension of the editor, not a standalone chatbot.
- "Read", "check", "show", "review", "summarize", "what do I have"
  → means use `read_editor`.
- "Write", "update", "fix", "draft", "improve", "rewrite", "change"
  → means use `update_editor`.
- Never guess or invent editor content. Always read it first.

YOUR TOOLS:
1. `read_editor` — Use FIRST for any task involving existing content
   (review, summarize, edit, fix, check word count, find typos, continue)
2. `update_editor` — Edit existing content via targeted line changes.
   Only return the changed lines, NOT the full document.
3. `writer` — Long-form writing (articles, blog posts, guides).
   Triggers plan → write → humanize pipeline automatically.
4. `search_agent` — Quick web lookups for facts or references.
5. `deep_research` — Comprehensive research on a topic.
6. `scrape_url` — Fetch content from a URL.

BEHAVIOR RULES:
- Be concise. 1-3 sentences unless asked for more.
- Do not explain your actions. Just do the work and confirm briefly.
- No disclaimers, caveats, or unnecessary commentary.
- Prefer `writer` tool for "write about X" (it plans, drafts, humanizes).
- For editing/reviewing existing content: `read_editor` → `update_editor`.
- Never hallucinate editor content.

{editor_change_guidelines}

When relevant, draw on these memories about the user's writing style:
{memory_context}
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
        memory_context=memory_context,
        editor_change_guidelines=EDITOR_CHANGE_GUIDELINES,
        MARKDOWN_RULES=MARKDOWN_RULES,
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
        "Extract stable, reusable facts about the USER only.\n\n"
        "EXTRACT:\n"
        "- Personal facts (name, age, location, job, company)\n"
        "- Projects or work they are doing\n"
        "- Skills, tools, or tech they use\n"
        "- Preferences, opinions, or dislikes\n"
        "- Habits or routines\n"
        "- Instructions for how the AI should behave\n"
        "- Relationships or team context\n\n"
        "WRITING STYLE (only if message is 30+ words or has a distinctive voice):\n"
        "- Note their tone, vocabulary, sentence structure, personality\n"
        "- Add a one-line replication note: how to write content in their exact style\n\n"
        "IGNORE:\n"
        "- Questions, greetings, filler\n"
        "- Statements about others, AI, or external topics\n"
        "- Anything inferred or guessed — only explicit signals\n\n"
        "Return a clean summary of extracted facts. "
        "If style profiling applies, append it at the end under 'Style:'. "
        "If nothing to extract, return 'NONE'."
    )
    extraction = await llm_classifier.ainvoke(
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

    response = await llm_secondary.ainvoke(
        [  # Use raw LLM
            {"role": "system", "content": SEARCH_QUERY_SYSTEM_PROMPT},
            {"role": "user", "content": topic},
        ]
    )

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

        response = await llm_secondary.ainvoke([HumanMessage(content=synthesis_prompt)])

        return {"final_research_report": response.content}
    except Exception as e:
        print(f"Error in finalize_research: {e}")
        # Fallback: simple concatenation
        return {"final_research_report": f"Research Report:\n\n{combined_content}"}


async def research_answer_node(state: ChatState):
    report = state["final_research_report"]
    response = await llm_secondary.ainvoke(
        [
            SystemMessage(
                content="Answer the user using the complete research. Do not call Tools."
            ),
            HumanMessage(content=report),
        ]
    )
    return {"messages": [response]}


# Planning Node
async def writer_planning_node(state: ChatState):
    my_topic = state.get("writer_topic")
    memory_lines = state.get("memories", [])
    memory_context = (
        "\n".join(f" - {m}" for m in memory_lines)
        if memory_lines
        else "No memories available."
    )
    plan: WritingPlan = await llm_WriterPlan.ainvoke(
        [
            {
                "role": "system",
                "content": f"""You are a expert content planner. Create a detailed writing plan.

User memories (use these for tone/style):
{memory_context}

Your plan must include:
1. A clear, engaging title
2. Sections with headings, purpose, paragraph count, code/image requirements
3. Estimated word count
4. Tone guidance based on user's writing style from memories

Be specific. For each section, say exactly what it should cover.
""",
            },
            {"role": "user", "content": f"Create a writing plan for: {my_topic}"},
        ]
    )
    print(plan)
    return {"writer_output": {"plan": plan.model_dump()}, "writer_iteration": 0}


# Write Content Node
async def write_content_node(state: ChatState):
    plan = state["writer_output"]["plan"]
    print("My Plan ----> ", plan)
    feedback = state["writer_output"].get("feedback", "")
    print("My Feedback ----> ", feedback)
    iteration = state["writer_iteration"]
    print("My Iteration ----> ", iteration)

    # prompt 1
    sections_prompt = ""
    for i, section in enumerate(plan["sections"], 1):
        sections_prompt += f"""
Section {i}:
- Heading: {section["heading"]}
- Purpose: {section["purpose"]}
- Paragraph Count: {section["paragraph_count"]}
- Code Blocks: {section["code_blocks"]}
- Image Suggestions: {section.get("image_suggestions", "None")}
"""
    # Prompt 2 - correction thingy
    correction_prompt = ""
    if feedback and iteration > 0:
        correction_prompt = f"""
PREVIOUS ITERATION FEEDBACK (fix these issues):
{feedback}
"""
    #  Prompt 3 - Main One
    system_content = f"""You are a expert content writer. Write in clear, engaging markdown.

TONE GUIDANCE:
{plan['tone_guidance']}

TARGET WORD COUNT: ~{plan['estimated_word_count']} words

WRITING RULES:
- Use proper markdown headings (# for title, ## for sections)
- Write in the user's voice per tone guidance above
- Include code blocks with language tags when required (```python, ```bash, etc.)
- Place `[Image: description]` placeholders where images are suggested
- Each paragraph should be 3-5 sentences
- Use bold for key terms, bullet points for lists
- Never include meta-commentary like "As mentioned above" or "In this section"
- End each section naturally — no concluding fluff
- Here is strict Mardown Rules {MARKDOWN_RULES}

SECTIONS TO WRITE:
{sections_prompt}
{correction_prompt}"""

    response = await llm_secondary.ainvoke(
        [
            {"role": "system", "content": system_content},
            {"role": "user", "content": f"Write the full content for: {plan['title']}"},
        ]
    )

    return {
        "writer_output": {"plan": plan, "draft": response.content, "feedback": feedback}
    }


# Humanize + Finalize Node


async def humanize_finalize_node(state: ChatState, runtime: Runtime[UserContext]):
    plan = state["writer_output"]["plan"]
    draft = state["writer_output"]["draft"]
    tone_guidance = plan.get(
        "tone_guidance", "Write in a clean & clear way in humanize 8 grade student tone"
    )
    user_id = runtime.context.user_id
    memory_namespace = ("memories", user_id)
    style_memories = await runtime.store.asearch(
        memory_namespace,
        query="writing style tone vocabulary voice preferences",
        limit=5,
    )
    memory_lines = [d.value["data"] for d in style_memories]
    all_memories = list[set(state.get("memories", []) + memory_lines)]
    memory_context = "\n".join(f"- {m}" for m in all_memories)
    humanize_prompt = f"""You are a style editor. Your job: rewrite the content below to match the user's voice.

USER'S WRITING STYLE (from memories):
{memory_context}

TONE GUIDANCE:
{tone_guidance}

RULES:
- Preserve all facts, code blocks, and structure from the original
- Adjust: sentence rhythm, word choice, formality level, paragraph length
- Use the user's typical vocabulary and phrasing patterns
- Keep markdown formatting intact (headings, code blocks, lists)
- Do not add or remove content sections — only adjust style
- Replace `[Image: ...]` placeholders with actual markdown image syntax if context is sufficient (use a placeholder URL like `https://placehold.co/800x400` if no real URL known)

ORIGINAL CONTENT:
{draft}

Return ONLY the rewritten content. No explanations, no meta-commentary."""
    response = await llm_secondary.ainvoke(
        [
            {"role": "system", "content": humanize_prompt},
            {"role": "user", "content": "Rewrite this in the user's voice."},
        ]
    )
    humanized = response.content

    # Checking Part
    check_prompt = f"""Evaluate this content:

PLAN:
Title: {plan['title']}
Sections: {[s['heading'] for s in plan['sections']]}

CONTENT:
{humanized[:4000]}

Check for:
1. STYLE MATCH: Does it match the user's voice from these memories?
{memory_context}
2. COMPLETENESS: Are all planned sections present and fleshed out?
3. ISSUES: Missing sections, broken markdown, placeholder images not resolved, orphaned references, incomplete sentences."""

    check: StyleCheck = await llm_style_check.ainvoke(
        [{"role": "system", "content": check_prompt}]
    )
    if check.style_match_score < 6 or check.completeness_score < 6:
        check.should_loop = True
        if not check.issues:
            check.issues = ["Quality scores too low. Needs revision."]
    # Chekcking Here ---
    if check.should_loop and state["writer_iteration"] < 2:
        return {
            "writer_output": {
                "plan": plan,
                "draft": humanized,
                "feedback": (
                    "\n".join(check.issues)
                    if check.issues
                    else "Style score too low. Make it match the user's voice better."
                ),
            },
            "writer_iteration": state["writer_iteration"] + 1,
        }
    return {
        "writer_output": {
            "plan": plan,
            "draft": draft,
            "humanized": humanized,
            "feedback": "",
        },
        "editor_content": humanized,
        "writer_requested": False,
        "writer_iteration": 0,
    }


