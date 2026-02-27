<img width="4000" height="1000" alt="image" src="https://github.com/user-attachments/assets/ae54c116-5ef8-4280-9025-bab22740bcd0" />

# ShotQuality Agent

An AI agent system for analyzing NBA shot tracking data using natural language queries. Built with LangGraph, this multi-agent system retrieves game data from the ShotQuality API, performs basketball analytics, and generates visualization code. Supports any LLM provider compatible with LangChain's `init_chat_model`.

## Project Structure

```
ShotQualityAgent/
├── MainAgent.py                 # Agent definitions, LangGraph workflow, and main entrypoint
├── test_coder_integration.py    # Streamlit demo app
├── tools.py                     # LangChain tool wrappers around the ShotQuality API
├── data.py                      # Raw API calls, player location fetching, and tracking feature calculations
├── requirements.txt             # Python dependencies
├── langgraph.json               # LangGraph configuration
└── .env.example                 # Environment variable template
```

## Overview

This project implements three specialized agents that work together to provide basketball performance insights:

**Loader Agent** - Fetches player, team, game, and shot tracking data from the ShotQuality API. Handles natural language queries like "Steph Curry's game against the Lakers in 2023", presents matching games for user selection, and loads full play-by-play tracking data once a game is confirmed. Uses the smaller configured LLM.

**Analyzer Agent** - Performs professional-level analysis on shot tracking data. Evaluates defender positioning, spacing metrics, shot quality, and provides coaching-level insights grounded in the geometric tracking features. Uses the larger configured LLM.

**Coder Agent** - Generates clean, executable Python code for data visualization. Creates matplotlib, seaborn, or plotly code to produce shot charts, heatmaps, and other basketball analytics visualizations. Uses the larger configured LLM.

## How It Works

### Layered Architecture

- **`data.py`** handles all direct API communication — paginated player location fetches, play-by-play data, and the calculation of advanced geometric tracking features (defender angles, spacing triangles, bounding box metrics, etc.).
- **`tools.py`** wraps those functions as LangChain `@tool` definitions so the agents can call them. Tools include `get_player_id`, `get_team_id`, `get_competition_seasons`, `get_games`, `get_full_tracking_data`, `get_nba_season`, and `search_duckduckgo` (for resolving ambiguous player/team references).
- **`MainAgent.py`** defines the agent prompts, constructs the LangGraph `StateGraph`, and wires up the routing logic.

### Routing

Two router functions control the flow:

- **`followup_router`** — fires on every new user message from `START`. Decides whether to send the request to the Loader (new data), Analyzer (follow-up question on existing data), or Coder (visualization request).
- **`router`** — handles transitions mid-workflow: tool call dispatch, looping the Loader until data is confirmed, handing off to the Analyzer once tracking data is loaded, and terminating after Analyzer or Coder responses.

Conversation state is persisted across turns using LangGraph's `InMemorySaver` checkpointer, so follow-up questions retain full context.

### Loader Workflow

1. `get_player_id` → resolve player
2. `search_duckduckgo` → find the player's team for the requested season
3. `get_team_id` → resolve team
4. `get_competition_seasons` → get the competition season ID
5. `get_games` → fetch the season schedule for that team
6. Present matching games to the user and wait for selection
7. `get_full_tracking_data` → load the confirmed game's shot + tracking data

The tracking data includes advanced geometric features: shooter distance and angle to basket, defender distances and positioning angles, teammate spacing metrics, bounding box contest proxies, and historical defensive prior probabilities.

## Usage

### Streamlit Demo App

The easiest way to interact with the agent is via the Streamlit chat interface:

```bash
streamlit run test_coder_integration.py
```

This launches a chat UI in your browser with:
- A persistent chat window that maintains full conversation context across turns
- A sidebar with example queries and a button to clear chat history
- Streaming agent responses rendered in real time

### Terminal (CLI)

Run the agent directly from the terminal:

```bash
python3 MainAgent.py
```

Example queries:
- "Can you analyze Devin Booker's shots in game 16 of his 2022-23 season?"
- "How did the defenders guard him on three-pointers?"
- "Create a shot chart for this data"
- "Show me LeBron's game against the Warriors in 2023"

## Environment Variables

Copy `.env.example` to `.env` and fill in your values:

```
LLM_KEY=           # API key for your LLM provider
SHOTQUALITY_API_KEY=  # ShotQuality API key
LANGSMITH_API_KEY=    # LangSmith API key (for tracing)
SMALL_MODEL_NAME=     # Model name for the Loader agent (e.g. mistral-small-latest)
LARGE_MODEL_NAME=     # Model name for the Analyzer and Coder agents (e.g. mistral-large-latest)
MODEL_PROVIDER=       # LangChain model provider string (e.g. mistral, openai, anthropic)
```

## Requirements

- Python 3.8+
- ShotQuality API key
- LLM API key (any provider supported by LangChain's `init_chat_model`)
- Dependencies listed in `requirements.txt` (includes `streamlit` for the demo app)
