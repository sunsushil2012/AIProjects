Standardizing LLM Interaction with MCP Servers

Model Context Protocol, or MCP, is an open protocol that standardizes how applications provide context to LLMs. In other words it provides a unified framework for LLM based applications to connect to connect to data sources, get context, use tools, and execute standard prompts.

Setup and Installation

1. Clone the Repo

git clone https://github.com/sunsushil2012/AIProjects/aem-assistant
cd aem-assistant

2. run below command

python3 database.py

3. Create the Virtual Environment and Install Packages

# Using uv (recommended)
uv venv
source .venv/bin/activate  # On macOS/Linux
# OR
.venv\Scripts\activate     # On Windows

# Install dependencies
uv sync

4. Run the Client & Server

python client.py mcp_server.py

5. Please add .env file at root level 
   
   add OPENAI_API_KEY= <Add OpenAI key here>


If using pip as package manager

you can use the following command to install the required packages:

you can skip above step 3 and run below command

pip install -r requirements.txt





