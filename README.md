# ai_agent_project

Source - https://diamantai.substack.com/p/your-first-ai-agent-simpler-than 

# 🧠 Multi-Step Text Analysis Pipeline with LangChain & LangGraph

This project demonstrates a modular text analysis pipeline using [LangChain](https://python.langchain.com/) and [LangGraph](https://github.com/langchain-ai/langgraph). The pipeline:

1. **Classifies** input text (News, Blog, Research, Other)
2. **Extracts entities** (Person, Organization, Location)
3. **Summarizes** the text in one sentence

It leverages OpenAI’s GPT model (via `ChatOpenAI`) for each step.

---

## 🚀 Quick Start

### 1. **Install dependencies**

```bash
pip install langchain langgraph langchain-openai python-dotenv
```

### 2. **Environment setup**

Create a `.env` file with your OpenAI API key:

```
OPENAI_API_KEY=sk-...
```

Load environment variables in your code:

```python
from dotenv import load_dotenv
import os

load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_API_KEY')
```

---

## 🏗️ **Pipeline Structure**

### **State Definition**

```python
from typing import TypedDict, List

class State(TypedDict):
    text: str
    classification: str
    entities: List[str]
    summary: str
```

Defines the data passed between pipeline steps.

---

### **Initialize LLM**

```python
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
```

---

### **Node Functions**

#### 1. **Classification Node**

```python
def classification_node(state: State):
    prompt = PromptTemplate(...)
    message = HumanMessage(content=prompt.format(text=state["text"]))
    classification = llm.invoke([message]).content.strip()
    return {"classification": classification}
```

#### 2. **Entity Extraction Node**

```python
def entity_extraction_node(state: State):
    prompt = PromptTemplate(...)
    message = HumanMessage(content=prompt.format(text=state["text"]))
    entities = llm.invoke([message]).content.strip().split(", ")
    return {"entities": entities}
```

#### 3. **Summarization Node**

```python
def summarization_node(state: State):
    prompt = PromptTemplate(...)
    message = HumanMessage(content=prompt.format(text=state["text"]))
    summary = llm.invoke([message]).content.strip()
    return {"summary": summary}
```

---

### **Build the Workflow Graph**

```python
from langgraph.graph import StateGraph, END

workflow = StateGraph(State)
workflow.add_node("classification_node", classification_node)
workflow.add_node("entity_extraction", entity_extraction_node)
workflow.add_node("summarization", summarization_node)

workflow.set_entry_point("classification_node")
workflow.add_edge("classification_node", "entity_extraction")
workflow.add_edge("entity_extraction", "summarization")
workflow.add_edge("summarization", END)
```

---

### **Compile and Run**

```python
app = workflow.compile()

sample_text = """..."""
state_input = {"text": sample_text}
result = app.invoke(state_input)
```

---

### **Output**

```python
print("Classification:", result["classification"])
print("\nEntities:", result["entities"])
print("\nSummary:", result["summary"])
```

---

## 💡 **Why use this pipeline?**

- **Modular:** Each step is a separate function—easy to extend or modify.
- **Clear data flow:** LangGraph makes the workflow explicit and visualizable.
- **Reusable:** Add or swap steps as needed for your use case.

---

