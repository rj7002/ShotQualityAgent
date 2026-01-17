from datetime import datetime
import os
from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain.chat_models import init_chat_model
from langsmith import traceable
from langchain_core.messages import BaseMessage
from langchain_community.tools import DuckDuckGoSearchRun
from langgraph.graph import StateGraph, START,END
from langgraph.graph.message import add_messages
from typing import Sequence, TypedDict, Annotated
from langgraph.prebuilt import ToolNode
from tools import get_nba_season_f, get_player_id_f, get_team_id_f, get_competition_seasons_f, get_games_f, get_full_tracking_data_f
from langgraph.checkpoint.memory import InMemorySaver  
import functools
from langchain_core.prompts import ChatPromptTemplate
from nba_api.stats.endpoints import PlayerIndex
load_dotenv()

#LOAD ENVIRONMENT VARIABLES
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGSMITH_API_KEY"] = os.getenv("LANGSMITH_API_KEY")
os.environ["LANGCHAIN_PROJECT"] = "shotquality-agent"

#CREATE TOOLS
@tool
def get_nba_season(subtract):
    """
    Returns the current NBA season in YYYY-YY format.
    DON'T USE THIS TOOL IF THE USER HAS ALREADY GIVEN A NUMERIC NUMBER FOR THE SEASON
    ONLY USE THIS TOOL when a user asks about a non numeric term for a season like "this season", "last season",
    or relative NBA season references.
    If the user says "this season" or any reference to the current season without explicitly saying the number, subtract should be 0
    If the user says "last season", "past season", etc, then subtract should be 1
    And so forth depending on how the user says like it could be 5 seasons ago and then subtract would be 5
    """
    year = get_nba_season_f(subtract)
    return year

@tool
def get_player_id(name):
    """Get a player's id using the api from their name"""
    df = get_player_id_f(name)
    return df.to_json()

@tool
def get_team_name(name,season):
    """Get a player's team name using nba_api and the start season year"""
    players = PlayerIndex(season=f"{season}-{str(season+1)[-2:]}").get_data_frames()[0]
    players['player_name'] = players['FIRST_NAME'] + ' ' + players['LAST_NAME']
    players['team_name'] = players['TEAM_CITY'] + ' ' + players['TEAM_NAME']
    matched = players[players['player_name'].str.lower() == name.lower()]
    return matched['team_name'].iloc[0] if not matched.empty else "Team not found"

@tool
def get_team_id(name):
    """Get a team's id using the api from their name"""
    df = get_team_id_f(name)
    return df.to_json()


@tool
def get_competition_seasons(name, year):
    """
    Get the competition_season_id for a competition.
    The year parameter should be the start year so if someone says 2024 season you should interpret that as 2024-2025 season
    NBA → 'NBA', College → 'NCAAM'
    """
    df = get_competition_seasons_f(name,year)
    return df.to_json()

@tool
def get_games(compid, teamid):
    """Get games for a competition"""
    df = get_games_f(compid,teamid)
    return df.to_json()

@tool
def get_full_tracking_data(gameid,playerid):
    """Get full tracking data using gameid"""
    df = get_full_tracking_data_f(gameid,playerid)
    return df.to_json()

@tool("search_duckduckgo")
def search_duckduckgo(query: str):
    """
    Search the web for an answer or relevant information using DuckDuckGo search engine.
    This tool should be used to find player names, team names, competition details, or any other relevant information if a user gives an input that is unclear or incomplete.
    
    Examples:
    - If a user just says "Tell me about LeBron", use this to find his full name "LeBron James"
    - If a user describes a player as "the number 1 pick in the 2023 draft", use this to find "Victor Wembanyama"
    - If you need to find what team a player was on in a specific season, use the query format:
      "What was {{player_name}}'s team in the {{season}} NBA season"
      For example: "What was Giannis Antetokounmpo's team in the 2021-22 NBA season"
    
    Overall this tool is meant to help you fill in any missing details or context needed to use the ShotQuality API tools effectively.
    Use this tool whenever you need to clarify or find additional information about players, teams, or competitions.
    """
    search = DuckDuckGoSearchRun()
    return search.invoke(query)

# SYSTEM_PROMPT = open("system_prompt.txt").read()

LOADER_PROMPT = """
You are a data loader. Call tools to retrieve shot tracking data.

WORKFLOW:
1. get_player_id(player_name) → extract player_id
   IMPORTANT: If multiple player_ids are returned, ALWAYS use the FIRST one (index 0) as it is the most complete/accurate entry.
2. search_duckduckgo(query) → Find the player's team for the season
   Use the query format: "What was {{player_name}}'s team in the {{season}} NBA season"
   For example: "What was Giannis Antetokounmpo's team in the 2021-22 NBA season"
3. get_team_id(team_name) → extract team_id
YOU MUST GET THE PLAYER's TEAM's team_id to use for get_games
4. get_competition_seasons(competition_name="NBA", year=YYYY) → extract competition_season_id
   IMPORTANT: year should be the START year. For "2021-22 season" use year=2021
   - For "last season" use get_nba_season first
5. get_games(competition_season_id, team_id) → you will get a list of games

AFTER STEP 5:
- Parse the games JSON to identify which games match the user's request
- If the user mentioned a specific opponent (e.g., "against the Lakers"), filter for those games by looking at either the home_team or away_team fields
- If the user mentioned a specific date (e.g., "on March 5th, 2023"), filter for that date using the date field which is in MM-DD-YYYY format
- If the user mentioned a specific game number (e.g., "game 30 of the season"), use that to select the correct game from the filtered list (note that game numbers are 1-indexed)
- If multiple games still match the user's criteria, present the list of matching games to the user and ask them to select one by its game number:
- Present the matching games to the user in a clear format with:
  * Game number (index + 1)
  * Date
  * Home team vs Away team
  * Ask the user: "Which game would you like to analyze? Please respond with the game number."
- WAIT for user to select a game number
- DO NOT proceed to get_full_tracking_data until user confirms

6. After user selects game number, call get_full_tracking_data(game_id, player_id) → DONE

After get_full_tracking_data completes, STOP. Do not call more tools or write responses.
"""

ANALYZER_PROMPT = """
You are ShotQuality Analyst AI, a professional basketball analytics assistant.

You will receive tracking data from the data loader in the conversation history.
The data is in JSON format and contains play-by-play shot data with tracking features.

CRITICAL: You have NO tools available. Do NOT attempt to call any tools.
- If you see an empty JSON object {{}} from get_full_tracking_data, this means NO DATA is available for that game
- If no data is available, politely inform the user that tracking data is not available for that specific game
- Suggest they try a different game from the list provided earlier

IMPORTANT: If the user asks for a plot, chart, graph, or any visualization:
- Tell them you'll generate the visualization code for them
- Do NOT attempt to create the visualization yourself
- The system will automatically hand off to the coding assistant

Your job is to analyze player and team performance using this ShotQuality play-by-play data.
You think like a film-room analyst, not a fan. 

Your goal is to produce insights that would be useful to a coach, analyst, or front office.
ONLY USE THE DATA THAT IS PROVIDED TO YOU in the tool results from previous messages.

Make sure to also use the extra tracking data from the feature store variable in your analysis if the data is available.
This is what each feature in the feature store variable means:

Closest Defender Driving Layup Prior: Historical probability that the closest defender allows a made driving layup, reflecting effectiveness at contesting shots off the drive.
Closest Defender Standing Layup Prior: Historical probability that the closest defender allows a made standing layup, reflecting rim-protection ability on non-driving finishes.
Angle with the Second-Closest Defender: Geometric angle between the shooter and the basket with the second-closest defender as the vertex, indicating whether the defender is positioned in front of, to the side of, or away from the shot path.
Distance to the Closest Defender (Feet): Distance between the shooter and the nearest defender at the moment of the shot, representing on-ball defensive pressure.
Height of the Closest Defender (Inches): Physical height of the nearest defender, relevant for contesting shots vertically.
Weight of the Closest Defender (Pounds): Physical mass of the nearest defender, relevant for strength and contact on drives.
Number of Defenders Closer to the Basket: Count of defenders positioned closer to the basket than the shooter at shot time, indicating help defense and rim congestion.
Cosine of the Angle with the Closest Defender: Directional measure of whether the closest defender is between the shooter and the basket, to the side, or behind the shooter.
Distance to the Second-Closest Defender (Feet): Distance between the shooter and the second-closest defender, representing secondary help proximity.
Height of the Second-Closest Defender (Inches): Physical height of the second-closest defender.
Weight of the Second-Closest Defender (Pounds): Physical mass of the second-closest defender.
Cosine of the Angle with the Second-Closest Defender: Directional measure of the second defender’s positioning relative to the shooter and basket.
Height of the Closest Defender’s Bounding Box (Pixels): Apparent on-screen height of the closest defender, used as a proxy for posture, arm extension, and contest intensity.
Height of the Third-Closest Defender’s Bounding Box (Pixels): Apparent on-screen height of the third defender, indicating additional vertical presence near the play.
Height of the Second-Closest Defender’s Bounding Box (Pixels): Apparent on-screen height of the second defender.
Difference in Bounding Box Height with the Closest Defender (Pixels): Shooter bounding-box height minus closest defender bounding-box height, indicating relative vertical contest.
Difference in Bounding Box Height with the Second-Closest Defender (Pixels): Shooter bounding-box height minus second defender bounding-box height, indicating relative contest from help defense.

MORE TRACKING FEATURES DEFINITIONS (Advanced Geometric Metrics from Player Location Data):

SHOOTER FEATURES:
- shooter_distance_to_basket: Euclidean distance (in feet) from shooter location to basket at time of shot. Lower values indicate closer proximity to the rim.
- shooter_angle_to_basket: Angle (in degrees) from shooter to basket. 0° = directly in line with basket, higher values indicate sideline shots.
- num_defenders_tracked: Count of defenders with location data for this play. Indicates completeness of defensive tracking.

DEFENDER FEATURES:
- closest_defender_distance: Distance (in feet) from shooter to nearest defender. Lower values indicate tighter defensive coverage.
- second_closest_defender_distance: Distance to the second closest defender. Helps capture multi-defender pressure patterns.
- closest_defender_angle_to_shooter: Angle (degrees) formed by the triangle of closest defender–shooter–basket. Smaller angles mean defender is directly between shooter and basket (strong contest).
- second_closest_defender_angle_to_shooter: Same as above for second closest defender.
- avg_defender_dist_from_shooter: Mean distance of all defenders to shooter. Higher values indicate looser overall defense.
- shooter_2def_triangle_area: Area of triangle (in square feet) formed by shooter and two closest defenders. Larger areas suggest more space despite defensive coverage.
- angle_at_shooter_between_two_closest_defenders: Angle (degrees) at shooter formed by lines to the two closest defenders. Smaller angles indicate clustered defenders; larger angles indicate spread coverage.
- angle_between_three_closest_defenders: Angle (degrees) formed by the three closest defenders (used if >2 defenders). Helps measure defensive formation and coverage geometry.

TEAMMATE FEATURES:
- num_teammates_tracked: Count of teammates with location data for this play. Indicates completeness of offensive tracking.
- closest_teammate_distance: Distance to nearest teammate. Very low values may indicate screens or congestion.
- avg_teammate_distance_from_shooter: Mean distance from shooter to all teammates. Higher values indicate better offensive spacing.
- shooter_2tm_triangle_area: Area of triangle formed by shooter and two closest teammates. Indicates spacing and lane creation.
- angle_at_shooter_between_two_closest_teammates: Angle (degrees) at shooter formed by lines to two closest teammates. Smaller angles indicate clustered teammates; larger angles indicate open spacing.

COMBINED FEATURES:
- total_players_tracked: Total count of all tracked players (shooter + defenders + teammates). Maximum is 10. Higher values indicate more complete tracking.

IMPORTANT NOTES ABOUT TRACKING DATA:
- Not all features are available for every shot (depends on tracking availability)
- Shooter features are present for 97%+ of shots
- Defender features only exist when defenders are tracked (check num_defenders_tracked)
- Multi-defender metrics (e.g., defender_spread, closest_two_defenders) only exist when 2+ defenders are tracked
- Teammate features only exist when teammates are tracked (check num_teammates_tracked)
- Always reference num_defenders_tracked and num_teammates_tracked to understand data completeness

Overall, your breakdowns should be grounded in the data provided and numbers/statistics should always be highlighted to support your insights.
You must use the data provided to answer any questions that the user asks whether it be about shot quality, how a player was defended, or any other aspect of their performance.
Don't just regurgitate the information you received, instead, analyze it and provide insights based on the data.
Don't focus on specific shots but rather the overall trends and patterns you see in the data.
MAKE SURE YOU ONLY FOCUS ON WHAT THE USER ASKED ABOUT AND DO NOT PROVIDE EXTRA INFORMATION.
FOR EXAMPLE: if the user asks about how defenders guarded a player, only focus on the defensive aspects and do not provide information about offensive aspects.
Make sure your analysis uses numbers and statistics from the data to back up your insights. Make sure to use all the tracking data such as distances, areas, angles, etc. provided.
"""

def create_agent(llm, tools, system_message):
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_message),
            ("placeholder", "{messages}")
        ]
    )
    return prompt | llm.bind_tools(tools)

class AgentState(TypedDict):
    messages : Annotated[Sequence[BaseMessage], add_messages]
    sender : str
    pending_games : dict  # Store games waiting for user selection
    selected_game_id : int  # Store the confirmed game_id
    player_id : int  # Store player_id for tracking data

def agent_node(state, agent, name):
    result = agent.invoke(state)
    result.name = name
    return {
        "messages" : [result],
        "sender" : name
    }
llm = init_chat_model(model="mistral-small-2506",model_provider='mistralai',api_key=os.getenv("MISTRAL_KEY"), temperature=0.3, timeout=120)
llm2 = init_chat_model(model="mistral-large-2512",model_provider='mistralai',api_key=os.getenv("MISTRAL_KEY"), temperature=0.3, timeout=120)
loader_agent = create_agent(
    llm, 
    [get_nba_season, get_competition_seasons, get_games, get_player_id, get_team_id, get_full_tracking_data, search_duckduckgo],
    system_message=LOADER_PROMPT
)

loader_node = functools.partial(agent_node, agent=loader_agent, name="Loader")

analyzer_agent = create_agent(
    llm2,
    tools=[],
    system_message=ANALYZER_PROMPT
)

analyzer_node = functools.partial(agent_node, agent=analyzer_agent, name="Analyzer")

CODER_PROMPT = """
You are a Python coding assistant specialized in basketball data visualization.

You will receive shot tracking data from the conversation history (look for ToolMessage from get_full_tracking_data).
The data is in JSON format with columns like: shot_x, shot_y, made, action_type, period, minutes, seconds, 
and various tracking features (defender distances, angles, spacing metrics, etc.).

Your task is to write clean, executable Python code that:
1. Loads the tracking data from JSON (it's already in the conversation as a string)
2. Converts it to a pandas DataFrame
3. Creates the requested visualization using matplotlib, seaborn, or plotly
4. Saves the plot to a file (e.g., 'shot_chart.png')

IMPORTANT GUIDELINES:
- Start with necessary imports: import pandas as pd, import matplotlib.pyplot as plt, import json, etc.
- Parse the JSON data from the conversation: df = pd.DataFrame(json.loads(tracking_data_json))
- Use clear variable names and add helpful comments
- Make the visualization publication-quality with titles, labels, legends
- For shot charts, use NBA court dimensions (94 feet x 50 feet, basket at (5.25, 25) and (88.75, 25))
- Color-code by made/missed shots (green for made, red for missed is conventional)
- Include a legend and proper axis labels
- End with plt.savefig('filename.png') and plt.show()
- Use best practices: proper figure size, DPI, grid styling, etc.

EXAMPLE STRUCTURE:
```python
import pandas as pd
import matplotlib.pyplot as plt
import json

# Load the tracking data from the JSON string in conversation
tracking_data_json = '''<JSON_HERE>'''
df = pd.DataFrame(json.loads(tracking_data_json))

# Create visualization
fig, ax = plt.subplots(figsize=(12, 8))
# ... plotting code ...
plt.title('Shot Chart')
plt.xlabel('X Position (feet)')
plt.ylabel('Y Position (feet)')
plt.legend()
plt.tight_layout()
plt.savefig('shot_chart.png', dpi=300)
plt.show()
```

Respond ONLY with the Python code, no explanations before or after the code block.
"""

coder_agent = create_agent(
    llm2,  # Use the larger model for better code generation
    tools=[],
    system_message=CODER_PROMPT
)

coder_node = functools.partial(agent_node, agent=coder_agent, name="Coder")

from langchain_core.messages import ToolMessage

def router(state):
    last = state["messages"][-1]
    
    # Check if we have a ToolMessage from get_full_tracking_data - that means data is loaded
    if isinstance(last, ToolMessage):
        # Check the tool name directly from the ToolMessage
        if hasattr(last, "name") and last.name == "get_full_tracking_data":
            return "analyze"
        
        # If we got games data, check if we need to wait for user selection
        if hasattr(last, "name") and last.name == "get_games":
            # Continue to loader so it can present games to user
            return "continue"
        
        # Also check by looking back in message history for the tool call
        for msg in reversed(state["messages"]):
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                for tool_call in msg.tool_calls:
                    # tool_call can be a dict or object, handle both
                    tool_name = tool_call.get("name") if isinstance(tool_call, dict) else getattr(tool_call, "name", None)
                    if tool_name == "get_full_tracking_data":
                        return "analyze"
        # Other tool results go back to loader
        return "continue"

    if hasattr(last, "tool_calls") and last.tool_calls:
        return "call_tool"
    
    # If loader says ERROR, end
    if hasattr(last, "content") and "ERROR" in last.content:
        return "end"
    
    # If last message is from Coder, end so user can see the code
    if hasattr(last, "name") and last.name == "Coder":
        return "end"
    
    # If last message is from Analyzer (AI response with no tool calls), end for now (user can continue thread)
    if hasattr(last, "name") and last.name == "Analyzer":
        return "end"
    
    # If message is from Loader asking for game selection, wait for user response
    if hasattr(last, "name") and last.name == "Loader":
        content = last.content if isinstance(last.content, str) else str(last.content)
        if "which game" in content.lower() or "game number" in content.lower():
            # End and wait for user input
            return "end"

    # If we have content but no tool calls, continue
    return "continue"

def followup_router(state):
    """Route to load new data, analyze existing data, or visualize data based on conversation state"""
    from langchain_core.messages import HumanMessage, ToolMessage
    
    last = state["messages"][-1]
    
    if not isinstance(last, HumanMessage):
        return "end"
    
    # Handle content being either string or list
    content = last.content if isinstance(last.content, str) else str(last.content)
    content_lower = content.lower()
    
    # Check if previous message was asking for game selection
    if len(state["messages"]) >= 2:
        prev_msg = state["messages"][-2]
        if hasattr(prev_msg, "content"):
            prev_content = prev_msg.content if isinstance(prev_msg.content, str) else str(prev_msg.content)
            if "which game" in prev_content.lower() or "game number" in prev_content.lower():
                # User is responding to game selection prompt
                print("[ROUTER DEBUG] Routing to: load_data (game selection response)")
                return "load_data"
    
    # Check if we already have tracking data in the conversation
    has_tracking_data = any(
        isinstance(msg, ToolMessage) and msg.name == "get_full_tracking_data" 
        for msg in state["messages"]
    )
    
    # Debug output
    print(f"\n[ROUTER DEBUG] Has tracking data: {has_tracking_data}")
    print(f"[ROUTER DEBUG] Message count: {len(state['messages'])}")
    print(f"[ROUTER DEBUG] User query: {content[:100]}")
    
    # Check for visualization requests
    visualization_indicators = [
        "plot", "chart", "graph", "visualize", "visualization", "show me a",
        "create a plot", "make a chart", "draw", "diagram", "heat map", "heatmap",
        "shot chart", "scatter", "histogram", "bar chart", "line graph"
    ]
    
    wants_visualization = any(indicator in content_lower for indicator in visualization_indicators)
    
    if wants_visualization and has_tracking_data:
        print("[ROUTER DEBUG] Routing to: visualize (visualization request with data)")
        return "visualize"
    
    # Strong indicators that user wants NEW data (not analyzing existing)
    new_data_indicators = [
        "instead", "different", "another", "new", "other",
        "get me", "show me", "load", "fetch", "retrieve", "give me",
    ]
    
    # Strong indicators that user wants to ANALYZE existing data
    analyze_indicators = [
        "how did", "what about", "tell me more", "explain", "describe",
        "why", "when", "where", "his three", "his defense", "his shooting",
        "the defender", "the data", "these shots", "this game"
    ]
    
    # Check for strong signals
    wants_new_data = any(indicator in content_lower for indicator in new_data_indicators)
    wants_analysis = any(indicator in content_lower for indicator in analyze_indicators)
    
    print(f"[ROUTER DEBUG] Wants new data: {wants_new_data}")
    print(f"[ROUTER DEBUG] Wants analysis: {wants_analysis}")
    print(f"[ROUTER DEBUG] Wants visualization: {wants_visualization}")
    
    # If user explicitly wants new data, route to loader
    if wants_new_data and not wants_analysis:
        print("[ROUTER DEBUG] Routing to: load_data (explicit new data request)")
        return "load_data"
    
    # If we have tracking data and user wants analysis, route to analyzer
    if has_tracking_data and wants_analysis:
        print("[ROUTER DEBUG] Routing to: analyze_existing (follow-up analysis)")
        return "analyze_existing"
    
    # Default: if we have tracking data and it's ambiguous, analyze existing
    # Otherwise load new data
    if has_tracking_data and not wants_new_data:
        print("[ROUTER DEBUG] Routing to: analyze_existing (default with data)")
        return "analyze_existing"
    
    print("[ROUTER DEBUG] Routing to: load_data (default - no data or new request)")
    return "load_data"

workflow = StateGraph(AgentState)

workflow.add_node("Loader", loader_node)
workflow.add_node("Analyzer", analyzer_node)
workflow.add_node("Coder", coder_node)
tools_input = [get_nba_season, get_competition_seasons, get_games, get_player_id, get_team_id, get_full_tracking_data, search_duckduckgo]
toolnode = ToolNode(tools_input)
workflow.add_node("call_tool",toolnode)

# Entry point: decide if this is initial query, follow-up, or visualization request
workflow.add_conditional_edges(
    START,
    followup_router,
    {
        "load_data": "Loader",
        "analyze_existing": "Analyzer",
        "visualize": "Coder"
    }
)

workflow.add_conditional_edges(
    "Loader",
    router,
    {
        "continue": "Loader",
        "call_tool": "call_tool",
        "analyze": "Analyzer",
        "end": END
    }
)
workflow.add_conditional_edges(
    "Analyzer",
    router,
    {
        "continue" : END,
        "call_tool": "call_tool",
        "end" : END
    }
)
workflow.add_conditional_edges(
    "Coder",
    router,
    {
        "end": END
    }
)
workflow.add_conditional_edges(
    "call_tool",
    router,
    {
        "continue": "Loader",
        "analyze": "Analyzer",
        "end": END
    }
)
# workflow.add_conditional_edges(
#     "call_tool",
#     lambda x: x["sender"],
#     {
#         "Loader" : "Loader",
#         "Analyzer" : "Analyzer"
#     }
# )

# full_agent = workflow.compile()

full_agent = workflow.compile(checkpointer=InMemorySaver())

# def print_stream(stream):
#     for s in stream:
#         message = s["messages"][-1]
#         if isinstance(message, tuple):
#             print(message)
#         else:
#             message.pretty_print()
# user_input = input("USER: " )
# while user_input != "exit":
#     inputs = {
#     "messages": [
#         {"role": "user", "content": user_input}
#     ]
# }
#     config = {
#     "configurable": {
#         "thread_id": "1"
#     }
# }


#     print_stream(full_agent.stream(inputs, stream_mode="values", config=config))
#     user_input = input("USER: ")
