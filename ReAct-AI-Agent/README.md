================================================================================
ReAct-AI-Agent Framework Dashboard
================================================================================

This directory demonstrates a raw Python implementation of the ReAct (Reasoning and Acting) Agent architecture without external agent dependency frameworks (like LangChain or CrewAI). It uses native loops and regular expression parsing matching outputs from Groq LLMs.

DIRECTORY STRUCTURE:
- React_agent_Python.ipynb : Notebook mapping out prompt development and core loop design.
- app.py                   : Custom interactive multi-step trace Streamlit application.
- requirements.txt         : Packages needed for local deployment.
- .gitignore.txt           : Default git exclusion properties.

GETTING STARTED
1. Create and activate a python virtual environment:
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate

2. Install dependencies:
   pip install -r requirements.txt

3. Run the interactive dashboard:
   streamlit run app.py

4. Provide your Groq API Key within the application sidebar control frame to test out the tools.
