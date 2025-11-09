"""
Sample agent for Case Closed Challenge - Works with Judge Protocol
This agent runs as a Flask server and responds to judge requests.
"""

import os
from flask import Flask, request, jsonify
from collections import deque

app = Flask(__name__)

# Basic identity
PARTICIPANT = os.getenv("PARTICIPANT", "SampleParticipant")
AGENT_NAME = os.getenv("AGENT_NAME", "SampleAgent")

# Track game state
game_state = {
    "board": None,
    "agent1_trail": [],
    "agent2_trail": [],
    "agent1_length": 0,
    "agent2_length": 0,
    "agent1_alive": True,
    "agent2_alive": True,
    "agent1_boosts": 3,
    "agent2_boosts": 3,
    "turn_count": 0,
    "player_number": 1,
}


@app.route("/", methods=["GET"])
def info():
    """Basic health/info endpoint used by the judge to check connectivity."""
    return jsonify({"participant": PARTICIPANT, "agent_name": AGENT_NAME}), 200


@app.route("/send-state", methods=["POST"])
def receive_state():
    """Judge calls this to push the current game state to the agent server."""
    data = request.get_json()
    if not data:
        return jsonify({"error": "no json body"}), 400
    
    # Update our local game state
    game_state.update(data)
    
    return jsonify({"status": "state received"}), 200


@app.route("/send-move", methods=["GET"])
def send_move():
    """Judge calls this (GET) to request the agent's move for the current tick.
    
    Return format: {"move": "DIRECTION"} or {"move": "DIRECTION:BOOST"}
    """
    player_number = request.args.get("player_number", default=1, type=int)
    turn_count = game_state.get("turn_count", 0)
    
    # Get our current state
    if player_number == 1:
        my_trail = game_state.get("agent1_trail", [])
        my_boosts = game_state.get("agent1_boosts", 3)
        other_trail = game_state.get("agent2_trail", [])
    else:
        my_trail = game_state.get("agent2_trail", [])
        my_boosts = game_state.get("agent2_boosts", 3)
        other_trail = game_state.get("agent1_trail", [])
    
    # Simple decision logic
    move = decide_move(my_trail, other_trail, turn_count, my_boosts)
    
    return jsonify({"move": move}), 200


@app.route("/end", methods=["POST"])
def end_game():
    """Judge notifies agent that the match finished and provides final state."""
    data = request.get_json()
    if data:
        result = data.get("result", "UNKNOWN")
        print(f"\nGame Over! Result: {result}")
    return jsonify({"status": "acknowledged"}), 200


def decide_move(my_trail, other_trail, turn_count, my_boosts):
    """Simple decision logic for the agent.
    
    Strategy:
    - Move in a direction that doesn't immediately hit a trail
    - Use boost if we have them and it's mid-game (turns 30-80)
    """
    if not my_trail:
        return "RIGHT"
    
    # Get current head position and direction
    head = my_trail[-1] if my_trail else (0, 0)
    
    # Calculate current direction if we have at least 2 positions
    current_dir = "RIGHT"
    if len(my_trail) >= 2:
        prev = my_trail[-2]
        dx = head[0] - prev[0]
        dy = head[1] - prev[1]
        
        # Normalize for torus wrapping
        if abs(dx) > 1:
            dx = -1 if dx > 0 else 1
        if abs(dy) > 1:
            dy = -1 if dy > 0 else 1
        
        if dx == 1:
            current_dir = "RIGHT"
        elif dx == -1:
            current_dir = "LEFT"
        elif dy == 1:
            current_dir = "DOWN"
        elif dy == -1:
            current_dir = "UP"
    
    # Simple strategy: try to avoid trails, prefer continuing straight
    # Check available directions (not opposite to current)
    directions = ["UP", "DOWN", "LEFT", "RIGHT"]
    opposite = {"UP": "DOWN", "DOWN": "UP", "LEFT": "RIGHT", "RIGHT": "LEFT"}
    
    # Remove opposite direction
    if current_dir in opposite:
        try:
            directions.remove(opposite[current_dir])
        except ValueError:
            pass
    
    # Prefer current direction if still available
    if current_dir in directions:
        chosen_dir = current_dir
    else:
        # Pick first available
        chosen_dir = directions[0] if directions else "RIGHT"
    
    # Decide whether to use boost
    # Use boost in mid-game when we still have them
    use_boost = my_boosts > 0 and 30 <= turn_count <= 80
    
    if use_boost:
        return f"{chosen_dir}:BOOST"
    else:
        return chosen_dir


if __name__ == "__main__":
    # For development only. Port can be overridden with the PORT env var.
    port = int(os.environ.get("PORT", "5009"))
    print(f"Starting {AGENT_NAME} ({PARTICIPANT}) on port {port}...")
    app.run(host="0.0.0.0", port=port, debug=False)
