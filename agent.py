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
 
PARTICIPANT = "CaseClosed"
AGENT_NAME = "SmartTronAgent"

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
    
    def flood_fill(self, start: Tuple[int, int], board: List[List[int]]) -> Set[Tuple[int, int]]:
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
    
    def _calculate_safety_distance(self, pos: Tuple[int, int], board: List[List[int]]) -> int:
        """Calculate minimum distance to nearest obstacle."""
        safety = float('inf')
        x, y = pos
        
        for i in range(self.board_height):
            for j in range(self.board_width):
                if board[i][j] == 1:
                    dx = min(abs(x - j), self.board_width - abs(x - j))
                    dy = min(abs(y - i), self.board_height - abs(y - i))
                    safety = min(safety, dx + dy)
        
        return safety if safety != float('inf') else 0
    
    def _torus_distance(self, pos1: Tuple[int, int], pos2: Tuple[int, int]) -> int:
        """Calculate Manhattan distance with torus wrapping."""
        dx = min(abs(pos1[0] - pos2[0]), self.board_width - abs(pos1[0] - pos2[0]))
        dy = min(abs(pos1[1] - pos2[1]), self.board_height - abs(pos1[1] - pos2[1]))
        return dx + dy

    def _should_use_boost(self, my_pos: Tuple[int, int], opp_pos: Tuple[int, int], board: List[List[int]], boosts: int, turn_count: int) -> bool:
        """Decide whether to use a boost."""
        if boosts <= 0 or turn_count < 5:
            return False
            
        dist_to_opp = self._torus_distance(my_pos, opp_pos)
        space_around = len(self.pathfinder.flood_fill(my_pos, board))
        
        # Use boost if we have good amount of space and opponent is somewhat far
        return space_around > 100 and dist_to_opp > 3

    def get_move(self, state: dict, player_number: int) -> str:
        """Main decision-making function."""
        # Extract positions from state
        if player_number == 1:
            my_trail = state.get('agent1_trail', [])
            opp_trail = state.get('agent2_trail', [])
            my_boosts = state.get('agent1_boosts', 3)
        else:
            my_trail = state.get('agent2_trail', [])
            opp_trail = state.get('agent1_trail', [])
            my_boosts = state.get('agent2_boosts', 3)
        
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
        
        # Evaluate each valid move
        best_move = None
        best_score = float('-inf')
        
        for move in valid_moves:
            dx, dy = self.directions[move]
            new_x = (my_pos[0] + dx) % self.board_width
            new_y = (my_pos[1] + dy) % self.board_height
            new_pos = (new_x, new_y)
            
            # Create simulated board
            sim_board = [row[:] for row in board]
            sim_board[new_y][new_x] = 1
            
            score = 0
            
            # 1. Survival: accessible space
            accessible_space = len(self.pathfinder.flood_fill(new_pos, sim_board))
            score += accessible_space * 10
            
            # 2. Safety distance
            safety_dist = self._calculate_safety_distance(new_pos, sim_board)
            score += safety_dist * 2
            
            # 3. Opponent proximity - prefer staying at medium distance
            dist_to_opp = self._torus_distance(new_pos, opp_pos)
            if dist_to_opp < 2:  # Too close
                score -= 50
            elif dist_to_opp > 8:  # Too far
                score -= 20
            
            if score > best_score:
                best_score = score
                best_move = move
        
        best_move = best_move or random.choice(valid_moves)
        
        # Decide whether to use boost
        use_boost = self._should_use_boost(my_pos, opp_pos, board, my_boosts, turn_count)
        
        self.last_move = best_move
        return f"{best_move}:BOOST" if use_boost else best_move

# Initialize pathfinder and agent
pathfinder = Pathfinder()
tron_agent = TronAgent()


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
   
    # Get the move from our TronAgent
    move = tron_agent.get_move(state, player_number)
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
    
    def bfs(
        self, 
        start: Tuple[int, int], 
        goal: Tuple[int, int],
        board: List[List[int]]
    ) -> Optional[List[Tuple[int, int]]]:
        """Breadth-first search to find shortest path."""
        start = self._normalize_pos(start)
        goal = self._normalize_pos(goal)
        
        if start == goal:
            return [start]
        
        queue = deque([(start, [start])])
        visited = {start}
        
        while queue:
            (x, y), path = queue.popleft()
            
            for dx, dy in self.directions:
                new_pos = self._normalize_pos((x + dx, y + dy))
                
                if new_pos == goal:
                    return path + [goal]
                
                if new_pos not in visited and board[new_pos[1]][new_pos[0]] == 0:
                    visited.add(new_pos)
                    queue.append((new_pos, path + [new_pos]))
        
        return None
    
    def a_star(
        self, 
        start: Tuple[int, int], 
        goal: Tuple[int, int],
        board: List[List[int]]
    ) -> Optional[List[Tuple[int, int]]]:
        """A* pathfinding algorithm with torus distance."""
        from heapq import heappush, heappop
        
        def heuristic(pos: Tuple[int, int]) -> int:
            """Manhattan distance with torus wrapping."""
            dx = min(abs(pos[0] - goal[0]), self.width - abs(pos[0] - goal[0]))
            dy = min(abs(pos[1] - goal[1]), self.height - abs(pos[1] - goal[1]))
            return dx + dy
        
        start = self._normalize_pos(start)
        goal = self._normalize_pos(goal)
        
        if start == goal:
            return [start]
        
        open_set = [(0, start)]
        came_from = {}
        g_score = {start: 0}
        f_score = {start: heuristic(start)}
        visited = set()
        
        while open_set:
            current_f, current = heappop(open_set)
            
            if current in visited:
                continue
            
            visited.add(current)
            
            if current == goal:
                # Reconstruct path
                path = [current]
                while current in came_from:
                    current = came_from[current]
                    path.append(current)
                return path[::-1]
            
            x, y = current
            for dx, dy in self.directions:
                neighbor = self._normalize_pos((x + dx, y + dy))
                
                if board[neighbor[1]][neighbor[0]] == 0:
                    tentative_g = g_score.get(current, float('inf')) + 1
                    
                    if tentative_g < g_score.get(neighbor, float('inf')):
                        came_from[neighbor] = current
                        g_score[neighbor] = tentative_g
                        f_score[neighbor] = tentative_g + heuristic(neighbor)
                        heappush(open_set, (f_score[neighbor], neighbor))
        
        return None
    
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
        self.opponent_history = []
        self.pathfinder = Pathfinder(self.board_width, self.board_height)
        
    def get_move(self, state: dict, player_number: int) -> str:
        """
        Main decision-making function.
        
        Args:
            state: Dictionary containing game state
            player_number: 1 or 2 indicating which player we are
        
        Returns:
            str: Direction to move ('UP', 'DOWN', 'LEFT', 'RIGHT')
                 or 'DIRECTION:BOOST' to use a boost
        """
        # Extract positions from state
        if player_number == 1:
            my_trail = state.get('agent1_trail', [])
            opp_trail = state.get('agent2_trail', [])
            my_boosts = state.get('agent1_boosts', 3)
        else:
            my_trail = state.get('agent2_trail', [])
            opp_trail = state.get('agent1_trail', [])
            my_boosts = state.get('agent2_boosts', 3)
        
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
        best_move = self._evaluate_moves(my_pos, opp_pos, board, valid_moves, state)
        
        # Decide whether to use boost
        use_boost = self._should_use_boost(my_pos, opp_pos, board, my_boosts, turn_count)
        
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
        state: dict
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
            score += dist_to_opp
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
