# Chapter 02 — CrewAI Core Concepts

**Goal:** Understand the four building blocks of CrewAI — Agent, Task, Tool, and Crew — by writing and running tiny examples. You will NOT build the real agents yet. This chapter is purely conceptual with runnable code you write yourself.

**Time estimate:** 30–45 minutes

> **Rule for this chapter:** Type every snippet yourself. Do not copy-paste. The muscle memory of writing these patterns matters.

---

## The Four Building Blocks

```
Tool  ──►  Agent  ──►  Task  ──►  Crew
```

- A **Tool** gives an agent an ability (search the web, read a file, call an API)
- An **Agent** is an AI persona with a role, a goal, and optional tools
- A **Task** is a specific instruction given to an agent, with an expected output
- A **Crew** is the team — it holds agents and tasks and runs them in order

---

## Concept 1 — The Agent

An Agent in CrewAI has four required properties:

| Property | What it does |
|----------|-------------|
| `role` | Short title — "Senior Dermatologist", "Research Analyst" |
| `goal` | What this agent is trying to achieve |
| `backstory` | Personality and expertise context — this shapes how the LLM responds |
| `llm` | Which language model powers this agent |

**Create a scratch file to experiment in** (do not put this in your `agents/` folder yet):

```powershell
# In your project root
New-Item scratch.py -ItemType File
```

Write this in `scratch.py`:

```python
from crewai import Agent, LLM

# Point CrewAI at your local Ollama instance
# LiteLLM (which CrewAI uses) needs the "ollama/" prefix
llm = LLM(
    model="ollama/qwen2.5:7b",
    base_url="http://localhost:11434",
)

# Define an agent
greeter = Agent(
    role="Friendly Greeter",
    goal="Introduce yourself clearly and warmly",
    backstory=(
        "You are a helpful assistant at a medical clinic. "
        "You greet patients warmly and explain your purpose clearly."
    ),
    llm=llm,
    verbose=True,   # prints the agent's thinking to the terminal
)

print("Agent created successfully:", greeter.role)
```

Run it:
```powershell
python scratch.py
```

**What to observe:** It should print the agent's role. No LLM call happens yet — agents need Tasks before they run.

**Experiment:** Change `verbose=True` to `verbose=False`. Note what changes. Change it back.

---

## Concept 2 — The Task

A Task tells an agent what to do RIGHT NOW. It has:

| Property | What it does |
|----------|-------------|
| `description` | The actual instruction — be specific |
| `expected_output` | What format/content you want back — guides the model |
| `agent` | Which agent handles this task |

Add this to `scratch.py` (keep the agent code, add below it):

```python
from crewai import Task

introduce = Task(
    description=(
        "Introduce yourself to a new patient named Ravi who has just arrived "
        "at the clinic for a skin consultation. Mention what you do and how you will help."
    ),
    expected_output=(
        "A warm 2-3 sentence introduction addressed directly to Ravi."
    ),
    agent=greeter,
)

print("Task created:", introduce.description[:50], "...")
```

Run again. Still no LLM call — tasks also need a Crew to execute.

> **Key insight:** The `expected_output` field is not just documentation. CrewAI includes it in the prompt sent to the LLM as an instruction about what to produce. Write it precisely — vague expected outputs produce vague responses.

---

## Concept 3 — The Crew

The Crew assembles agents and tasks and actually runs them.

Add this to `scratch.py`:

```python
from crewai import Crew, Process

crew = Crew(
    agents=[greeter],
    tasks=[introduce],
    process=Process.sequential,  # run tasks one after another in order
    verbose=True,
)

# This line actually calls the LLM
result = crew.kickoff()

print("\n--- RESULT ---")
print(result)
```

Run it:
```powershell
python scratch.py
```

**What to observe:**
- CrewAI prints a formatted panel showing the agent thinking through the task
- The final output is printed after `--- RESULT ---`
- It should be a personalised greeting to "Ravi"

**Experiment — change the patient name:**  
Edit the task `description` to say a different name. Run again. The output should change accordingly.

---

## Concept 4 — Process Types

CrewAI has two main process types:

### `Process.sequential`
Tasks run one after another. Each task's output is available to the next task as context. **Use this when tasks have a clear dependency chain.**

```
Task 1 → Task 2 → Task 3 → Final Output
```

### `Process.hierarchical`
A "manager" agent coordinates other agents. The manager can delegate, ask for re-runs, and synthesise. **This is what our Orchestrator will use.**

```
Manager Agent
  ├── delegates to Agent A → gets result
  ├── delegates to Agent B → gets result
  ├── sees conflict, asks Agent A to revise
  └── synthesises final output
```

For hierarchical process you must specify a `manager_llm`:
```python
crew = Crew(
    agents=[agent_a, agent_b],
    tasks=[task_a, task_b],
    process=Process.hierarchical,
    manager_llm=LLM(model="ollama/qwen2.5:7b", base_url="http://localhost:11434"),
    verbose=True,
)
```

> **Our project uses sequential for the analysis phase and we give the Orchestrator access to all previous outputs for the final synthesis. This keeps the architecture simple and predictable.**

---

## Concept 5 — Tools

A Tool is a Python function wrapped so that an agent can call it. The agent reads the tool's name and description, decides whether to use it, and passes arguments to it.

Write a minimal tool in `scratch.py`:

```python
from crewai.tools import BaseTool
from pydantic import BaseModel, Field

# Define what arguments the tool accepts
class MultiplyInput(BaseModel):
    a: int = Field(description="First number")
    b: int = Field(description="Second number")

class MultiplyTool(BaseTool):
    name: str = "multiply"
    description: str = (
        "Multiplies two integers together. "
        "Use this when you need to calculate a product."
    )
    args_schema: type[BaseModel] = MultiplyInput

    def _run(self, a: int, b: int) -> str:
        result = a * b
        return f"The product of {a} and {b} is {result}"

# Create an agent that has the tool
calculator = Agent(
    role="Calculator",
    goal="Perform arithmetic calculations accurately",
    backstory="You are precise and always use the multiply tool for multiplication.",
    llm=llm,
    tools=[MultiplyTool()],   # pass an instance, not the class
    verbose=True,
)

calc_task = Task(
    description="What is 7 multiplied by 13?",
    expected_output="The exact integer result of the multiplication.",
    agent=calculator,
)

calc_crew = Crew(
    agents=[calculator],
    tasks=[calc_task],
    process=Process.sequential,
    verbose=True,
)

result = calc_crew.kickoff()
print("\n--- CALC RESULT ---")
print(result)
```

Run it. You should see the agent decide to use the `multiply` tool, call it with `a=7, b=13`, get `91`, and return it.

**What to observe carefully:**
- The agent's "thought" process is printed (it says something like "I should use the multiply tool")
- The tool is called with structured arguments
- The result is incorporated into the agent's response

> **This pattern — define input schema, inherit BaseTool, implement `_run` — is exactly how you will build the ImageAnalysisTool and PubMedSearchTool in later chapters.**

---

## Concept 6 — Task Context (Agent-to-Agent Communication)

In CrewAI, agents share information through **task context**. When you set `context=[some_task]` on a task, the output of `some_task` is automatically injected into the current task's prompt.

```python
task_b = Task(
    description="Summarise the greeting in one sentence.",
    expected_output="One sentence summary.",
    agent=some_agent,
    context=[introduce],   # task_b receives task introduce's output
)
```

This is how your lesion agents will pass their findings to the Research Agent, and how the Research Agent passes evidence to the Orchestrator.

---

## Concept 7 — Structured JSON Output

For our agents, we want outputs like:
```json
{"lesion_colour": "erythematous red", "reason": "..."}
```

Not prose. You enforce this two ways:

**1. In `expected_output`** — be explicit:
```
Return ONLY a valid JSON object with keys "lesion_colour" and "reason". No prose, no markdown fences.
```

**2. Using Pydantic output parsing** — CrewAI can auto-parse into a Pydantic model:
```python
from pydantic import BaseModel

class ColourOutput(BaseModel):
    lesion_colour: str
    reason: str

task = Task(
    description="...",
    expected_output="JSON with lesion_colour and reason",
    agent=colour_agent,
    output_pydantic=ColourOutput,  # CrewAI validates and parses automatically
)
```

You will use `output_pydantic` for all your lesion agents. It ensures downstream agents always get clean structured data, not unpredictable prose.

---

## Clean Up

Delete `scratch.py` — it was only for learning:
```powershell
Remove-Item scratch.py
```

---

## Checkpoint ✅

You should now be able to answer these questions without looking at notes. If you can't, re-read the relevant section:

- [ ] What is the difference between an Agent and a Task?
- [ ] What does `verbose=True` on an Agent do?
- [ ] What is the difference between `Process.sequential` and `Process.hierarchical`?
- [ ] Why does `expected_output` matter beyond documentation?
- [ ] How does one task get access to another task's output?
- [ ] What are the three things you must implement to create a custom Tool?
- [ ] What does `output_pydantic` do on a Task?

---

*Next → `03_OLLAMA_CONNECTION.md`*
