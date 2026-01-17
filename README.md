# ShotQuality Agent

An AI agent system for analyzing NBA shot tracking data using natural language queries. Built with LangGraph and Mistral AI, this multi-agent system retrieves game data from the ShotQuality API, performs basketball analytics, and generates visualization code.

## Overview

This project implements three specialized agents that work together to provide basketball performance insights:

**Loader Agent** - Fetches player, team, game, and shot tracking data from the ShotQuality API. Handles natural language queries like "Steph Curry's game against the Lakers in 2023" and presents matching games for user selection.

**Analyzer Agent** - Performs professional-level analysis on shot tracking data. Evaluates defender positioning, spacing metrics, shot quality, and provides coaching-level insights based on geometric tracking features.

**Coder Agent** - Generates clean Python code for data visualization. Creates matplotlib, seaborn, or plotly code to produce shot charts, heatmaps, and other basketball analytics visualizations.

## How It Works

Users interact with the system through natural language queries. The system intelligently routes requests to the appropriate agent based on context and intent. It maintains conversation state to handle follow-up questions and supports queries ranging from specific game requests to follow-up analysis and visualization requests.

The Loader Agent follows a structured workflow to retrieve data: it identifies the player, determines their team for the specified season, fetches the season's games, presents matching games to the user, and loads the selected game's tracking data.

The tracking data includes advanced geometric features such as shooter distance to basket, defender positioning and angles, teammate spacing metrics, and historical defensive effectiveness measures.

## Usage

Run the interactive agent:
```bash
python3 MainAgent.py
```

Example queries:
- "Can you analyze Devin Booker's shots in game 16 of his 2022-23 season?"
- "How did the defenders guard him on three-pointers?"
- "Create a shot chart for this data"
- "Show me LeBron's game against the Warriors in 2023"

## Requirements

- Python 3.8+
- ShotQuality API key
- LLM API key
- Dependencies listed in requirements.txt
