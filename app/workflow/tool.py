from typing import Annotated, Any, Literal
from langchain_core.messages import ToolMessage
from langchain_core.tools import InjectedToolCallId, tool
from langchain_nvidia_ai_endpoints import ChatNVIDIA, NVIDIAEmbeddings
from langchain_openrouter import ChatOpenRouter
from langgraph.prebuilt import ToolNode, InjectedState
from langgraph.types import Command
from pydantic import BaseModel, ConfigDict, Field
from tinyfish import TinyFish,AsyncTinyFish

from app.config import settings


client = TinyFish(api_key=settings.TINYFISH_API_KEY)


class ReadEditorInput(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    state: Annotated[dict[str, Any], InjectedState()] | None = Field(
        default=None,
        description="Injected graph state. Not provided by the model.",
    )


MARKDOWN_RULES = """\
Markdown formatting rules the editor understands:

INLINE
- Bold: **text** or __text__
- Italic: *text* or _text_
- Underline: <u>text</u>
- Strikethrough: ~~text~~
- Inline code: `text`
- Link: [text](url)

BLOCK
- Headings: # H1, ## H2, ### H3, #### H4, ##### H5, ###### H6
- Blockquote: > text
- Ordered list: 1. item, 2. item
- Bullet list: - item, * item, or + item
- Code block: ```language\\ncode\\n```
- Horizontal rule: ---, ***, or ___
- Image: ![alt](url)

NOTES
- Bold (**) takes precedence over italic (*) when both could match.
- Use a backslash (\\) to escape special characters when needed.
"""


@tool
def multiply(a: int, b: int) -> int:
    """Multiply two numbers."""
    return a * b


@tool(args_schema=ReadEditorInput)
def read_editor(
    state: Annotated[dict[str, Any], InjectedState],
) -> str:
    """Read the full current content of the user's markdown editor.
    Use this whenever you need to see what the user is currently writing
    before you can help them (summarize, edit, review, continue, etc.).
    Returns the editor content as a single markdown string. If the editor
    is empty, returns an empty string.
    """
    content = state.get("editor_content", "") or ""
    if not content.strip():
        return "The editor is currently empty."
    return content


@tool
def write_editor(
    content: str,
    tool_call_id: Annotated[str, InjectedToolCallId] = "",
) -> Command:
    """Write markdown content to the user's editor, REPLACING the existing
    content. Use this when the user asks you to write, draft, update, fix,
    or replace what is in their editor.

    The `content` argument MUST be valid markdown using the supported syntax:

    - Inline: **bold**, *italic*, ~~strike~~, `inline code`, [link](url),
      <u>underline</u>
    - Block: # Heading, > blockquote, - bullet, 1. ordered,
      ```code block```, --- horizontal rule, ![alt](image-url)

    Always pass the FULL new content for the editor. If you need to preserve
    the current content, call `read_editor` first and include the existing
    text in `content` as appropriate.
    """
    confirmation = f"Wrote {len(content)} characters to the editor."

    return Command(
        update={
            "editor_content": content,
            "messages": [
                ToolMessage(confirmation, tool_call_id=tool_call_id),
            ],
        }
    )


@tool
async def search_agent(query: str) -> list:
    """Search the web for information. Always use this tool whenever you need
    current information or real-time data. Give a short query as input (max 400 tokens)."""
    response =  client.search.query(query=query, location="US")
    top_urls = [r.url for r in response.results[:3]]
    pages =  client.fetch.get_contents(urls=top_urls, format="markdown")
    return pages.results

# In tool.py - replace deep_research tool
@tool
def deep_research(
    topic: str,
    tool_call_id: Annotated[str, InjectedToolCallId] = "",
) -> Command:
    """Trigger deep research on a topic. Use when user asks for comprehensive analysis."""
    return Command(
        update={
            "topic": topic,  # Set the topic for research_topic_node
            "research_requested": True,
            "messages": [ToolMessage(f"Let me do a Deep research on: {topic}", tool_call_id=tool_call_id)]
        },
    )
@tool
def writer(
    topic: str,
    tool_call_id: Annotated[str, InjectedToolCallId] = "",
) -> Command:
    """Call this tool when you want to write content. just provide a clear topic"""
    return Command(
        update={
            "writer_topic": topic,  # Set the topic for research_topic_node
            "writer_requested": True,
            "messages": [ToolMessage(f"Let me write about {topic}", tool_call_id=tool_call_id)]
        },
    )


my_tools = [multiply, search_agent, read_editor, deep_research, writer,write_editor]

llm = ChatNVIDIA(
    model="stepfun-ai/step-3.5-flash",
    api_key=settings.NVIDIA_API_KEY,
    temperature=1,
    top_p=0.95,
    max_completion_tokens=16384,
    model_kwargs={"enable_thinking": True, "reasoning_budget": 3000},
)
# llm_classifier = ChatOpenRouter(
#     model="nex-agi/nex-n2-pro:free",
#     api_key=settings.OPENROUTER_API_KEY,
#     temperature=0.3,
#     reasoning={"effort": "low"},
#     top_p=0.95,
#     max_completion_tokens=2048,
# )
llm_secondary = ChatNVIDIA(
    model="stepfun-ai/step-3.5-flash",
    api_key=settings.NVIDIA_API_KEY,
    temperature=0.3,
    max_completion_tokens=8192,
    model_kwargs={"enable_thinking": False},
)
llm_classifier = ChatNVIDIA(
    model="nvidia/llama-3.3-nemotron-super-49b-v1",  # 8B params is plenty for classification
    api_key=settings.NVIDIA_API_KEY,
    temperature=0,   # 0 = deterministic, no randomness. We want consistent YES/NO.
    max_completion_tokens=2048,  # only need a short JSON
)

# A small, fast model for extracting the actual fact
llm_extractor = ChatNVIDIA(
    model="nvidia/llama-3.3-nemotron-super-49b-v1",
    api_key=settings.NVIDIA_API_KEY,
    temperature=0,
    max_completion_tokens=200,  # a fact is short
)

EMBEDDING_DIMS = 1024
embeddings = NVIDIAEmbeddings(
    model="nvidia/nv-embedqa-e5-v5",
    api_key=settings.NVIDIA_API_KEY,
    truncate="END",
)

llm_with_tool = llm.bind_tools(my_tools)
tool_node = ToolNode(my_tools)

# Some shitty stuff ----

class RememberDecision(BaseModel):
    """The classifier's answer. Forces a clean yes/no + reason."""
    should_remember: bool = Field(
        description="True if the user is sharing a fact, preference, or instruction to remember"
    )
    reason: str = Field(
        description="One short sentence explaining why"
    )

llm_classifier_remeber_structured = llm_classifier.with_structured_output(RememberDecision)

class OnlyHandyReasearchTopic(BaseModel):
    urls:list[str]=Field(
        description="List of handy URLs from the data to research"
    )
llm_OnlyHandyReasearchTopic = llm_classifier.with_structured_output(OnlyHandyReasearchTopic)

class SummarizedPageContent(BaseModel):
    summary:str=Field(
        description="Summarized content of the site"
    )
llm_SummarizedPageContent = llm_classifier.with_structured_output(SummarizedPageContent)

# In tool.py - modify ResearchTopic model
class ResearchTopic(BaseModel):
    topics: list[str] = Field(description="List of 2-4 research search queries")

llm_ResearchTopic = llm_classifier.with_structured_output(ResearchTopic)

# Writer Plan schema
class WritingSection(BaseModel):
    heading: str = Field(description="Section heading/title")
    purpose: str = Field(description="What this section should achieve")
    paragraph_count: int = Field(description="How many paragraphs")
    code_blocks: bool = Field(description="Whether code snippets are needed")
    image_suggestions: str | None = Field(default=None, description="Image description if needed")

class WritingPlan(BaseModel):
    title: str = Field(description="Proposed title of the piece")
    sections: list[WritingSection] = Field(description="List of sections")
    estimated_word_count: int = Field(description="Target total word count")
    tone_guidance: str = Field(description="Tone, voice, and style instructions")
    
llm_WriterPlan = llm_classifier.with_structured_output(WritingPlan)

# Humanize Schema
class StyleCheck(BaseModel):
    style_match_score: int = Field(description="1-10")
    completeness_score: int = Field(description="1-10")
    issues: list[str] = Field(description="Specific issues")
    should_loop: bool = Field(description="True if content needs another iteration")

llm_style_check = llm_classifier.with_structured_output(StyleCheck)

# update editor

class EditorChange(BaseModel):
    line: int = Field(description="1-indexed block-level line number")
    type: Literal["replace", "delete", "insert"] = Field(
        description="replace=swap content at line, delete=remove line, insert=add new content AFTER this line"
    )
    content: str = Field(description="New content (for replace/insert). Empty string for delete.")


class UpdateEditorInput(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    changes: list[EditorChange] = Field(
        description="List of line-level changes. ONLY include lines that changed."
    )
    state: Annotated[dict[str, Any], InjectedState()] | None = Field(
        default=None,
        description="Injected graph state. Not provided by the model.",
    )