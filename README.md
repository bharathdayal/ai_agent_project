# ai_agent_project

Source - https://diamantai.substack.com/p/your-first-ai-agent-simpler-than 

What is this code doing?
This code builds a multi-step text analysis pipeline using LangChain and LangGraph. The pipeline does:

1️⃣ Classifies the text into categories (News, Blog, Research, Other).
2️⃣ Extracts entities (Person, Organization, Location).
3️⃣ Summarizes the text in one short sentence.

It uses OpenAI’s GPT model (via ChatOpenAI) to perform each step.

✅ Libraries and setup
python
Copy
Edit
import os
from typing import TypedDict, List
from langgraph.graph import StateGraph, END
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage
from dotenv import load_dotenv
os, dotenv: To manage environment variables and set up your OpenAI API key securely.

TypedDict, List: Type hinting to define what data will flow between steps.

LangGraph: A new LangChain feature for building graph-based workflows.

PromptTemplate & ChatOpenAI: Used to interact with OpenAI’s language models.

✅ Environment setup
python
Copy
Edit
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_API_KEY')
Loads your .env file.

Sets OPENAI_API_KEY for authentication.

✅ Define state (data structure)
python
Copy
Edit
class State(TypedDict):
    text: str
    classification: str
    entities: List[str]
    summary: str
State defines what data each node in the workflow expects or returns.

Helps manage the data consistently throughout the pipeline.

✅ Initialize LLM
python
Copy
Edit
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
Uses GPT-4o-mini.

temperature=0 means deterministic (less random) outputs.

✅ Define nodes
1️⃣ Classification node
python
Copy
Edit
def classification_node(state: State):
    prompt = PromptTemplate(...)
    message = HumanMessage(content=prompt.format(text=state["text"]))
    classification = llm.invoke([message]).content.strip()
    return {"classification": classification}
Classifies the text as News, Blog, Research, or Other.

Returns the classification label.

2️⃣ Entity extraction node
python
Copy
Edit
def entity_extraction_node(state: State):
    prompt = PromptTemplate(...)
    message = HumanMessage(content=prompt.format(text=state["text"]))
    entities = llm.invoke([message]).content.strip().split(", ")
    return {"entities": entities}
Extracts entities as a comma-separated list and converts to Python list.

3️⃣ Summarization node
python
Copy
Edit
def summarization_node(state: State):
    prompt = PromptTemplate(...)
    message = HumanMessage(content=prompt.format(text=state["text"]))
    summary = llm.invoke([message]).content.strip()
    return {"summary": summary}
Creates a one-sentence summary of the text.

✅ Build the workflow graph
python
Copy
Edit
workflow = StateGraph(State)

workflow.add_node("classification_node", classification_node)
workflow.add_node("entity_extraction", entity_extraction_node)
workflow.add_node("summarization", summarization_node)
Adds each node to the graph, assigning unique names.

Define edges
python
Copy
Edit
workflow.set_entry_point("classification_node")
workflow.add_edge("classification_node", "entity_extraction")
workflow.add_edge("entity_extraction", "summarization")
workflow.add_edge("summarization", END)
Specifies the order of execution:

Start with classification_node.

Then entity_extraction.

Then summarization.

Finally, stop (END).

✅ Compile and run
python
Copy
Edit
app = workflow.compile()
Finalizes the workflow so it can be executed.

Sample input and invocation
python
Copy
Edit
sample_text = """..."""
state_input = {"text": sample_text}
result = app.invoke(state_input)
Passes sample text as input state.

Runs through the graph and returns a state with classification, entities, and summary.

✅ Output
python
Copy
Edit
print("Classification:", result["classification"])
print("\nEntities:", result["entities"])
print("\nSummary:", result["summary"])
Prints the results from each step of the pipeline.

✅ 💡 Summary: Why is this useful?
✅ You can process text step by step in a modular and clear way.
✅ Easy to modify or extend (e.g., add more analysis steps).
✅ Uses LangGraph to create explicit data flow (like a visual graph).
✅ Clean separation of logic per node (classification, extraction, summarization).
