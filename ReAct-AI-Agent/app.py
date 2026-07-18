import os
import re
import streamlit as st
from groq import Groq

# -----------------------------------------------------------------------------
# 1. Page Configuration & Title
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="ReAct AI Agent Dashboard",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 ReAct AI Agent Dashboard")
st.markdown(
    """
    Explore the **Reasoning and Acting (ReAct)** pattern using Groq and standard Python.
    No heavy agent frameworks—just structured prompting and programmatic execution loops.
    """
)

# -----------------------------------------------------------------------------
# 2. Sidebar Configuration (API Key & Settings)
# -----------------------------------------------------------------------------
st.sidebar.header("⚙️ Settings")
groq_api_key = st.sidebar.text_input("Enter Groq API Key:", type="password")
model_choice = st.sidebar.selectbox(
    "Select Model:",
    ["llama-3.3-70b-versatile", "llama-3-1-70b-versatile"]
)
max_steps = st.sidebar.slider("Max Reasoning Steps", min_value=1, max_value=10, value=6)

# -----------------------------------------------------------------------------
# 3. Define Tools & Core Logic
# -----------------------------------------------------------------------------
def calculator(expression: str) -> str:
    """Evaluate a math expression and return the result as a string."""
    try:
        # Simple sandbox configuration
        return str(eval(expression, {"__builtins__": {}}, {}))
    except Exception as e:
        return f"Error: {e}"

def get_weather(city: str) -> str:
    """Return current weather for a city (mocked)."""
    data = {
        "chennai": "32°C, humid, partly cloudy",
        "bangalore": "24°C, pleasant, light rain",
        "delhi": "28°C, hazy",
        "mumbai": "30°C, humid",
    }
    return data.get(city.lower(), f"No weather data for {city}")

def word_count(text: str) -> str:
    """Count whitespace-separated words in a string."""
    return str(len(text.split()))

# Mapping dictionary for tool orchestration
TOOLS = {
    "calculator": calculator,
    "get_weather": get_weather,
    "word_count": word_count
}

TOOL_DESCRIPTIONS = """
- calculator(expression: str) -> str
    Evaluate a math expression. Example: calculator("23 * 47")
- get_weather(city: str) -> str
    Return current weather for a city. Example: get_weather("Chennai")
- word_count(text: str) -> str
    Count words in text. Example: word_count("hello world")
"""

SYSTEM_PROMPT = f"""You are a ReAct agent that solves problems step by step.

Tools available:
{TOOL_DESCRIPTIONS}

Format (follow exactly):

Thought: <reasoning>
Action: <tool_name>
Action Input: <input string>

After Action, STOP. The system replies with:

Observation: <result>

Continue with another Thought/Action, or finish:

Thought: I now know the final answer.
Final Answer: <answer>

Rules:
- One Thought + Action per turn, then wait.
- Action must be one of: {list(TOOLS.keys())}
- Action Input is a plain string (no quotes).
- Never invent Observations.
"""

def parse_response(text: str):
    """Parse LLM output into ('final', answer) | ('action', name, input) | ('error', msg)."""
    if m := re.search(r"Final Answer:\s*(.+)", text, re.DOTALL):
        return ("final", m.group(1).strip())

    a = re.search(r"Action:\s*(.+)", text)
    i = re.search(r"Action Input:\s*(.+)", text)
    if a and i:
        return ("action", a.group(1).strip(), i.group(1).strip().strip('"').strip("'"))

    return ("error", "Could not parse Action or Final Answer.")

# -----------------------------------------------------------------------------
# 4. User Interaction & Execution Loop
# -----------------------------------------------------------------------------
user_question = st.text_input(
    "Ask the ReAct Agent a question:", 
    placeholder="e.g., What is 234*17, then minus 3? Or check weather in Bangalore."
)

if st.button("Run Agent", type="primary"):
    if not groq_api_key:
        st.error("Please enter your Groq API Key in the sidebar.")
    elif not user_question.strip():
        st.warning("Please type a valid question.")
    else:
        try:
            # Initialize client locally
            client = Groq(api_key=groq_api_key)
            
            # Application logs UI layout
            st.subheader("🪵 Execution Logs & Reasoning Loop")
            status_container = st.container()
            
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_question},
            ]
            
            final_ans = None
            
            for step in range(1, max_steps + 1):
                with status_container:
                    with st.expander(f"Step {step}: Reasoning Cycle", expanded=True):
                        # Request next block from LLM
                        resp = client.chat.completions.create(
                            model=model_choice, 
                            messages=messages, 
                            temperature=0,
                            stop=["Observation:"],
                        )
                        out = resp.choices[0].message.content.strip()
                        messages.append({"role": "assistant", "content": out})
                        
                        # Display LLM output
                        st.markdown("**🤖 Assistant Thinking / Action Execution:**")
                        st.code(out)
                        
                        parsed = parse_response(out)
                        
                        if parsed[0] == "final":
                            final_ans = parsed[1]
                            st.success(f"Final Answer Found!")
                            break
                        if parsed[0] == "error":
                            final_ans = f"Error: {parsed[1]}"
                            st.error(parsed[1])
                            break
                        
                        _, name, arg = parsed
                        if name in TOOLS:
                            try:
                                obs = TOOLS[name](arg)
                            except Exception as e:
                                obs = f"Error running {name}: {e}"
                        else:
                            obs = f"Error: unknown tool '{name}'. Available: {list(TOOLS.keys())}"
                        
                        st.markdown(f"**🔧 Tool Output (Observation) via `{name}`:**")
                        st.info(obs)
                        
                        messages.append({"role": "user", "content": f"Observation: {obs}"})
            
            if not final_ans and step == max_steps:
                final_ans = "Agent stopped: Maximum reasoning steps reached without final conclusion."
            
            # Display summary output
            st.markdown("---")
            st.subheader("✅ Final Output Summary")
            st.success(final_ans)
            
        except Exception as err:
            st.error(f"An unexpected error occurred: {err}")
