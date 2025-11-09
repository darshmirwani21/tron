import os
import uuid
from flask import Flask, request, jsonify
from threading import Lock
from collections import deque

from case_closed_game import Game, Direction, GameResult

# Flask API server setup
app = Flask(__name__)

GLOBAL_GAME = Game()
LAST_POSTED_STATE = {}

game_lock = Lock()
 
PARTICIPANT = "ParticipantX"
AGENT_NAME = "AgentX"


@app.route("/", methods=["GET"])
def info():
    """Basic health/info endpoint used by the judge to check connectivity.

    Returns participant and agent_name (so Judge.check_latency can create Agent objects).
    """
    return jsonify({"participant": PARTICIPANT, "agent_name": AGENT_NAME}), 200


def _update_local_game_from_post(data: dict):
    """Update the local GLOBAL_GAME using the JSON posted by the judge.

    The judge posts a dictionary with keys matching the Judge.send_state payload
    (board, agent1_trail, agent2_trail, agent1_length, agent2_length, agent1_alive,
    agent2_alive, agent1_boosts, agent2_boosts, turn_count).
    """
    with game_lock:
        LAST_POSTED_STATE.clear()
        LAST_POSTED_STATE.update(data)

        if "board" in data:
            try:
                GLOBAL_GAME.board.grid = data["board"]
            except Exception:
                pass

        if "agent1_trail" in data:
            GLOBAL_GAME.agent1.trail = deque(tuple(p) for p in data["agent1_trail"]) 
        if "agent2_trail" in data:
            GLOBAL_GAME.agent2.trail = deque(tuple(p) for p in data["agent2_trail"]) 
        if "agent1_length" in data:
            GLOBAL_GAME.agent1.length = int(data["agent1_length"])
        if "agent2_length" in data:
            GLOBAL_GAME.agent2.length = int(data["agent2_length"])
        if "agent1_alive" in data:
            GLOBAL_GAME.agent1.alive = bool(data["agent1_alive"])
        if "agent2_alive" in data:
            GLOBAL_GAME.agent2.alive = bool(data["agent2_alive"])
        if "agent1_boosts" in data:
            GLOBAL_GAME.agent1.boosts_remaining = int(data["agent1_boosts"])
        if "agent2_boosts" in data:
            GLOBAL_GAME.agent2.boosts_remaining = int(data["agent2_boosts"])
        if "turn_count" in data:
            GLOBAL_GAME.turns = int(data["turn_count"])


@app.route("/send-state", methods=["POST"])
def receive_state():
    """Judge calls this to push the current game state to the agent server.

    The agent should update its local representation and return 200.
    """
    data = request.get_json()
    if not data:
        return jsonify({"error": "no json body"}), 400
    _update_local_game_from_post(data)
    return jsonify({"status": "state received"}), 200


@app.route("/send-move", methods=["GET"])
def send_move():
    """Judge calls this (GET) to request the agent's move for the current tick.

    Query params the judge sends (optional): player_number, attempt_number,
    random_moves_left, turn_count. Agents can use this to decide.
    
    Return format: {"move": "DIRECTION"} or {"move": "DIRECTION:BOOST"}
    where DIRECTION is UP, DOWN, LEFT, or RIGHT
    and :BOOST is optional to use a speed boost (move twice)
    """
    player_number = request.args.get("player_number", default=1, type=int)

    with game_lock:
        state = dict(LAST_POSTED_STATE)   
        my_agent = GLOBAL_GAME.agent1 if player_number == 1 else GLOBAL_GAME.agent2
        boosts_remaining = my_agent.boosts_remaining
   
    # -----------------your code here-------------------
"""
Case Closed - Tron Lightbike Agent
Fixed version with proper Flask integration
"""

import os
from flask import Flask, request, jsonify
from threading import Lock
from collections import deque
from typing import List, Tuple, Optional, Dict, Set
import random

from case_closed_game import Game, Direction, GameResult

# Flask API server setup
app = Flask(__name__)

GLOBAL_GAME = Game()
LAST_POSTED_STATE = {}
game_lock = Lock()

PARTICIPANT = "YourTeamName"  # CHANGE THIS
AGENT_NAME = "SmartTronAgent"  # CHANGE THIS


class Pathfinder:
    """Pathfinding algorithms for grid navigation with torus wrapping."""
    
    def __init__(self, width: int = 20, height: int = 18):
        """Initialize pathfinder."""
        self.width = width
        self.height = height
        self.directions = [(0, -1), (0, 1), (-1, 0), (1, 0)]
    
    def _normalize_pos(self, pos: Tuple[int, int]) -> Tuple[int, int]:
        """Normalize position for torus wrapping."""
        return (pos[0] % self.width, pos[1] % self.height)
    
    def flood_fill(
        self, 
        start: Tuple[int, int], 
        board: List[List[int]]
    ) -> Set[Tuple[int, int]]:
        """Flood fill to find all connected empty cells."""
        start = self._normalize_pos(start)
        visited = set()
        queue = deque([start])
        visited.add(start)
        
        while queue:
            x, y = queue.popleft()
            
            for dx, dy in self.directions:
                new_pos = self._normalize_pos((x + dx, y + dy))
                
                if new_pos not in visited and board[new_pos[1]][new_pos[0]] == 0:
                    visited.add(new_pos)
                    queue.append(new_pos)
        
        return visited


class TronAgent:
    """Main agent class for Tron game decision-making."""
    
    def __init__(self):
        """Initialize the agent."""
        self.board_width = 20
        self.board_height = 18
        self.directions = {
            'UP': (0, -1),
            'DOWN': (0, 1),
            'LEFT': (-1, 0),
            'RIGHT': (1, 0)
        }
        self.direction_names = ['UP', 'DOWN', 'LEFT', 'RIGHT']
        self.last_move = None
        self.pathfinder = Pathfinder(self.board_width, self.board_height)
        
    def get_move(self, state: dict, player_number: int, boosts_remaining: int) -> str:
        """
        Main decision-making function.
        
        Args:
            state: Dictionary containing game state
            player_number: 1 or 2 indicating which player we are
            boosts_remaining: Number of boosts left
        
        Returns:
            str: Direction to move ('UP', 'DOWN', 'LEFT', 'RIGHT')
                 or 'DIRECTION:BOOST' to use a boost
        """
        # Extract positions from state
        if player_number == 1:
            my_trail = state.get('agent1_trail', [])
            opp_trail = state.get('agent2_trail', [])
        else:
            my_trail = state.get('agent2_trail', [])
            opp_trail = state.get('agent1_trail', [])
        
        if not my_trail or not opp_trail:
            return 'RIGHT'
        
        my_pos = tuple(my_trail[-1])
        opp_pos = tuple(opp_trail[-1])
        board = state.get('board', [[0]*20 for _ in range(18)])
        turn_count = state.get('turn_count', 0)
        
        # Get all valid moves
        valid_moves = self._get_valid_moves(my_pos, board)
        
        if not valid_moves:
            return random.choice(self.direction_names)
        
        # Evaluate moves and choose best
        best_move = self._evaluate_moves(my_pos, opp_pos, board, valid_moves, turn_count)
        
        # Decide whether to use boost
        use_boost = self._should_use_boost(my_pos, opp_pos, board, boosts_remaining, turn_count)
        
        self.last_move = best_move
        
        if use_boost:
            return f"{best_move}:BOOST"
        return best_move
    
    def _get_valid_moves(self, position: Tuple[int, int], board: List[List[int]]) -> List[str]:
        """Get all valid moves from current position."""
        valid = []
        x, y = position
        
        for direction, (dx, dy) in self.directions.items():
            new_x = (x + dx) % self.board_width  # Torus wrapping
            new_y = (y + dy) % self.board_height
            
            # Check if cell is empty (0 = empty, 1 = trail)
            if board[new_y][new_x] == 0:
                valid.append(direction)
        
        return valid
    
    def _evaluate_moves(
        self, 
        my_pos: Tuple[int, int], 
        opp_pos: Tuple[int, int],
        board: List[List[int]], 
        valid_moves: List[str],
        turn_count: int
    ) -> str:
        """Evaluate all valid moves and return the best one."""
        move_scores = {}
        
        for move in valid_moves:
            dx, dy = self.directions[move]
            new_x = (my_pos[0] + dx) % self.board_width
            new_y = (my_pos[1] + dy) % self.board_height
            new_pos = (new_x, new_y)
            
            # Create simulated board
            sim_board = [row[:] for row in board]
            sim_board[new_y][new_x] = 1
            
            score = 0
            
            # 1. Survival: accessible space (using pathfinder flood fill)
            accessible_space = len(self.pathfinder.flood_fill(new_pos, sim_board))
            score += accessible_space * 10
            
            # 2. Safety distance
            safety_dist = self._calculate_safety_distance(new_pos, sim_board)
            score += safety_dist * 2
            
            # 3. Opponent proximity
            dist_to_opp = self._torus_distance(new_pos, opp_pos)
            if dist_to_opp < 3:
                score -= 5
            elif dist_to_opp > 10:
                score += 2
            
            # 4. Territory control
            territory = self._calculate_territory_control(new_pos, opp_pos, sim_board)
            score += territory * 3
            
            # 5. Avoid corners early
            if self._is_corner(new_pos) and turn_count < 50:
                score -= 3
            
            move_scores[move] = score
        
        if move_scores:
            return max(move_scores, key=move_scores.get)
        
        return valid_moves[0] if valid_moves else 'UP'
    
    def _calculate_safety_distance(
        self, 
        position: Tuple[int, int], 
        board: List[List[int]]
    ) -> int:
        """Calculate minimum distance to nearest trail using BFS."""
        visited = set()
        queue = deque([(position, 0)])
        visited.add(position)
        
        while queue:
            pos, dist = queue.popleft()
            x, y = pos
            
            if board[y][x] == 1 and dist > 0:
                return dist
            
            for dx, dy in self.directions.values():
                new_x = (x + dx) % self.board_width
                new_y = (y + dy) % self.board_height
                new_pos = (new_x, new_y)
                
                if new_pos not in visited:
                    visited.add(new_pos)
                    queue.append((new_pos, dist + 1))
        
        return 10
    
    def _calculate_territory_control(
        self, 
        my_pos: Tuple[int, int], 
        opp_pos: Tuple[int, int],
        board: List[List[int]]
    ) -> int:
        """Calculate territory control difference using flood fill."""
        my_space = len(self.pathfinder.flood_fill(my_pos, board))
        opp_space = len(self.pathfinder.flood_fill(opp_pos, board))
        return my_space - opp_space
    
    def _should_use_boost(
        self,
        my_pos: Tuple[int, int],
        opp_pos: Tuple[int, int],
        board: List[List[int]],
        boosts_remaining: int,
        turn_count: int
    ) -> bool:
        """Decide whether to use a boost."""
        if boosts_remaining <= 0:
            return False
        
        # Use boost in mid-game when we have good space
        if 30 <= turn_count <= 100:
            my_space = len(self.pathfinder.flood_fill(my_pos, board))
            if my_space > 80:
                return True
        
        # Use boost if opponent is close and we need to escape
        dist = self._torus_distance(my_pos, opp_pos)
        if dist < 5 and boosts_remaining > 1:
            return True
        
        return False
    
    def _torus_distance(self, pos1: Tuple[int, int], pos2: Tuple[int, int]) -> int:
        """Calculate Manhattan distance with torus wrapping."""
        dx = min(abs(pos1[0] - pos2[0]), self.board_width - abs(pos1[0] - pos2[0]))
        dy = min(abs(pos1[1] - pos2[1]), self.board_height - abs(pos1[1] - pos2[1]))
        return dx + dy
    
    def _is_corner(self, position: Tuple[int, int]) -> bool:
        """Check if position is in a corner."""
        x, y = position
        corners = [
            (0, 0), (self.board_width - 1, 0),
            (0, self.board_height - 1), 
            (self.board_width - 1, self.board_height - 1)
        ]
        return position in corners


# Global agent instance
agent = TronAgent()


# Flask API Endpoints

@app.route("/", methods=["GET"])
def info():
    """Basic health/info endpoint used by the judge to check connectivity."""
    return jsonify({"participant": PARTICIPANT, "agent_name": AGENT_NAME}), 200


def _update_local_game_from_post(data: dict):
    """Update the local GLOBAL_GAME using the JSON posted by the judge."""
    with game_lock:
        LAST_POSTED_STATE.clear()
        LAST_POSTED_STATE.update(data)

        if "board" in data:
            try:
                GLOBAL_GAME.board.grid = data["board"]
            except Exception:
                pass

        if "agent1_trail" in data:
            GLOBAL_GAME.agent1.trail = deque(tuple(p) for p in data["agent1_trail"]) 
        if "agent2_trail" in data:
            GLOBAL_GAME.agent2.trail = deque(tuple(p) for p in data["agent2_trail"]) 
        if "agent1_length" in data:
            GLOBAL_GAME.agent1.length = int(data["agent1_length"])
        if "agent2_length" in data:
            GLOBAL_GAME.agent2.length = int(data["agent2_length"])
        if "agent1_alive" in data:
            GLOBAL_GAME.agent1.alive = bool(data["agent1_alive"])
        if "agent2_alive" in data:
            GLOBAL_GAME.agent2.alive = bool(data["agent2_alive"])
        if "agent1_boosts" in data:
            GLOBAL_GAME.agent1.boosts_remaining = int(data["agent1_boosts"])
        if "agent2_boosts" in data:
            GLOBAL_GAME.agent2.boosts_remaining = int(data["agent2_boosts"])
        if "turn_count" in data:
            GLOBAL_GAME.turns = int(data["turn_count"])


@app.route("/send-state", methods=["POST"])
def receive_state():
    """Judge calls this to push the current game state to the agent server."""
    data = request.get_json()
    if not data:
        return jsonify({"error": "no json body"}), 400
    _update_local_game_from_post(data)
    return jsonify({"status": "state received"}), 200


@app.route("/send-move", methods=["GET"])
def send_move():
    """Judge calls this (GET) to request the agent's move for the current tick."""
    player_number = request.args.get("player_number", default=1, type=int)

    with game_lock:
        state = dict(LAST_POSTED_STATE)   
        my_agent = GLOBAL_GAME.agent1 if player_number == 1 else GLOBAL_GAME.agent2
        boosts_remaining = my_agent.boosts_remaining
    
    try:
        # Call our intelligent agent
        move = agent.get_move(state, player_number, boosts_remaining)
        return jsonify({"move": move}), 200
    except Exception as e:
        print(f"Error in get_move: {e}")
        import traceback
        traceback.print_exc()
        # Fallback to safe move
        return jsonify({"move": "RIGHT"}), 200


@app.route("/end", methods=["POST"])
def end_game():
    """Judge notifies agent that the match finished and provides final state."""
    data = request.get_json()
    if data:
        _update_local_game_from_post(data)
        result = data.get("result", "UNKNOWN")
        print(f"\nGame Over! Result: {result}")
    return jsonify({"status": "acknowledged"}), 200


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5008"))
    print(f"Starting {AGENT_NAME} ({PARTICIPANT}) on port {port}...")
    app.run(host="0.0.0.0", port=port, debug=False)     
    # -----------------end code here--------------------
    return jsonify({"move": move}), 200


@app.route("/end", methods=["POST"])
def end_game():
    """Judge notifies agent that the match finished and provides final state.

    We update local state for record-keeping and return OK.
    """
    data = request.get_json()
    if data:
        _update_local_game_from_post(data)
    return jsonify({"status": "acknowledged"}), 200


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5008"))
    app.run(host="0.0.0.0", port=port, debug=True)
