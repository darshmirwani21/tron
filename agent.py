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
   
"""
Case Closed - Tron Lightbike Agent
An intelligent agent for competitive Tron gameplay.
"""

from collections import deque
from typing import List, Tuple, Optional, Dict, Set
import random


class TronAgent:
    """Main agent class for Tron game decision-making."""
    
    def __init__(self):
        """Initialize the agent."""
        self.board_width = 20
        self.board_height = 18
        self.directions = {
            'up': (0, -1),
            'down': (0, 1),
            'left': (-1, 0),
            'right': (1, 0)
        }
        self.direction_names = ['up', 'down', 'left', 'right']
        self.last_move = None
        self.opponent_history = []
        
    def get_move(self, state: dict) -> str:
        """
        Main decision-making function called by the game engine.
        
        Args:
            state: Dictionary containing game state information:
                - my_position: (x, y) tuple of current position
                - opponent_position: (x, y) tuple of opponent position
                - board: 2D list representing the board (0=empty, 1=wall/trail)
                - boosts: Available boosts (if any)
                - turns_remaining: Number of turns left
                - opponent_last_direction: Last direction opponent moved
        
        Returns:
            str: Direction to move ('up', 'down', 'left', 'right')
        """
        my_pos = tuple(state['my_position'])
        opp_pos = tuple(state['opponent_position'])
        board = state['board']
        opponent_last_dir = state.get('opponent_last_direction', None)
        
        # Update opponent history
        if opponent_last_dir:
            self.opponent_history.append(opponent_last_dir)
            if len(self.opponent_history) > 10:
                self.opponent_history.pop(0)
        
        # Get all valid moves
        valid_moves = self._get_valid_moves(my_pos, board)
        
        if not valid_moves:
            # No valid moves, return a random direction (shouldn't happen)
            return random.choice(self.direction_names)
        
        # Evaluate each move and choose the best one
        best_move = self._evaluate_moves(
            my_pos, opp_pos, board, valid_moves, state
        )
        
        self.last_move = best_move
        return best_move
    
    def _get_valid_moves(self, position: Tuple[int, int], board: List[List[int]]) -> List[str]:
        """
        Get all valid moves from current position.
        
        Args:
            position: Current (x, y) position
            board: 2D board representation
        
        Returns:
            List of valid direction names
        """
        valid = []
        x, y = position
        
        for direction, (dx, dy) in self.directions.items():
            new_x, new_y = x + dx, y + dy
            
            # Check bounds
            if new_x < 0 or new_x >= self.board_width or new_y < 0 or new_y >= self.board_height:
                continue
            
            # Check if cell is empty (0 = empty, 1 = wall/trail)
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
        """
        Evaluate all valid moves and return the best one.
        
        Uses multiple strategies:
        1. Survival: Avoid immediate danger
        2. Space calculation: Maximize accessible space
        3. Opponent blocking: Try to limit opponent's space
        4. Pathfinding: Find safe paths to open areas
        """
        move_scores = {}
        
        for move in valid_moves:
            dx, dy = self.directions[move]
            new_pos = (my_pos[0] + dx, my_pos[1] + dy)
            
            # Create a simulated board with this move
            sim_board = [row[:] for row in board]
            sim_board[new_pos[1]][new_pos[0]] = 1
            
            score = 0
            
            # 1. Survival score: How much space is accessible from this position?
            accessible_space = self._calculate_accessible_space(new_pos, sim_board)
            score += accessible_space * 10  # High weight on survival
            
            # 2. Safety score: Distance to nearest wall/trail
            safety_dist = self._calculate_safety_distance(new_pos, sim_board)
            score += safety_dist * 2
            
            # 3. Opponent proximity: Maintain safe distance (not too close, not too far)
            dist_to_opp = abs(new_pos[0] - opp_pos[0]) + abs(new_pos[1] - opp_pos[1])
            if dist_to_opp < 3:
                score -= 5  # Too close is dangerous
            elif dist_to_opp > 10:
                score += 2  # Too far means we're not controlling space
            
            # 4. Territory control: How much area can we claim?
            territory = self._calculate_territory_control(new_pos, opp_pos, sim_board)
            score += territory * 3
            
            # 5. Pathfinding: Can we reach open areas?
            path_score = self._evaluate_pathfinding(new_pos, sim_board)
            score += path_score * 5
            
            # 6. Opponent prediction: Try to block opponent's likely moves
            block_score = self._evaluate_blocking(new_pos, opp_pos, sim_board, state)
            score += block_score * 2
            
            # 7. Avoid corners early in game
            if self._is_corner(new_pos):
                score -= 3
            
            move_scores[move] = score
        
        # Return move with highest score
        if move_scores:
            best_move = max(move_scores, key=move_scores.get)
            return best_move
        
        # Fallback to first valid move
        return valid_moves[0] if valid_moves else 'up'
    
    def _calculate_accessible_space(
        self, 
        position: Tuple[int, int], 
        board: List[List[int]]
    ) -> int:
        """
        Calculate accessible space using BFS flood fill.
        
        Args:
            position: Starting position
            board: Board state
        
        Returns:
            Number of accessible cells
        """
        visited = set()
        queue = deque([position])
        visited.add(position)
        
        while queue:
            x, y = queue.popleft()
            
            for dx, dy in self.directions.values():
                new_x, new_y = x + dx, y + dy
                new_pos = (new_x, new_y)
                
                if (new_pos not in visited and 
                    0 <= new_x < self.board_width and 
                    0 <= new_y < self.board_height and
                    board[new_y][new_x] == 0):
                    visited.add(new_pos)
                    queue.append(new_pos)
        
        return len(visited)
    
    def _calculate_safety_distance(
        self, 
        position: Tuple[int, int], 
        board: List[List[int]]
    ) -> int:
        """
        Calculate minimum distance to nearest wall/trail.
        
        Args:
            position: Current position
            board: Board state
        
        Returns:
            Minimum distance to wall
        """
        visited = set()
        queue = deque([(position, 0)])
        visited.add(position)
        
        while queue:
            (x, y), dist = queue.popleft()
            
            # If we found a wall, return the distance
            if board[y][x] == 1 and dist > 0:
                return dist
            
            for dx, dy in self.directions.values():
                new_x, new_y = x + dx, y + dy
                new_pos = (new_x, new_y)
                
                if (new_pos not in visited and 
                    0 <= new_x < self.board_width and 
                    0 <= new_y < self.board_height):
                    visited.add(new_pos)
                    queue.append((new_pos, dist + 1))
        
        return 10  # Default safe distance
    
    def _calculate_territory_control(
        self, 
        my_pos: Tuple[int, int], 
        opp_pos: Tuple[int, int],
        board: List[List[int]]
    ) -> int:
        """
        Calculate territory control difference.
        
        Args:
            my_pos: Our position
            opp_pos: Opponent position
            board: Board state
        
        Returns:
            Difference in accessible territory (ours - theirs)
        """
        my_space = self._calculate_accessible_space(my_pos, board)
        
        # Simulate opponent's accessible space
        opp_board = [row[:] for row in board]
        my_space_from_opp = self._calculate_accessible_space(opp_pos, opp_board)
        
        return my_space - my_space_from_opp
    
    def _evaluate_pathfinding(
        self, 
        position: Tuple[int, int], 
        board: List[List[int]]
    ) -> float:
        """
        Evaluate pathfinding quality using A* heuristic.
        
        Args:
            position: Current position
            board: Board state
        
        Returns:
            Pathfinding score (higher is better)
        """
        # Find the center of largest open area
        open_areas = self._find_open_areas(board)
        if not open_areas:
            return 0.0
        
        # Find distance to nearest large open area
        min_dist = float('inf')
        for area_center, area_size in open_areas:
            dist = abs(position[0] - area_center[0]) + abs(position[1] - area_center[1])
            if dist < min_dist:
                min_dist = dist
        
        # Inverse distance (closer is better)
        return 10.0 / (min_dist + 1)
    
    def _find_open_areas(self, board: List[List[int]]) -> List[Tuple[Tuple[int, int], int]]:
        """
        Find centers and sizes of open areas on the board.
        
        Args:
            board: Board state
        
        Returns:
            List of (center_position, area_size) tuples
        """
        visited = set()
        areas = []
        
        for y in range(self.board_height):
            for x in range(self.board_width):
                if board[y][x] == 0 and (x, y) not in visited:
                    # BFS to find connected area
                    area_cells = []
                    queue = deque([(x, y)])
                    visited.add((x, y))
                    
                    while queue:
                        cx, cy = queue.popleft()
                        area_cells.append((cx, cy))
                        
                        for dx, dy in self.directions.values():
                            new_x, new_y = cx + dx, cy + dy
                            new_pos = (new_x, new_y)
                            
                            if (new_pos not in visited and 
                                0 <= new_x < self.board_width and 
                                0 <= new_y < self.board_height and
                                board[new_y][new_x] == 0):
                                visited.add(new_pos)
                                queue.append(new_pos)
                    
                    if area_cells:
                        # Calculate center
                        avg_x = sum(cell[0] for cell in area_cells) // len(area_cells)
                        avg_y = sum(cell[1] for cell in area_cells) // len(area_cells)
                        areas.append(((avg_x, avg_y), len(area_cells)))
        
        # Sort by size (largest first)
        areas.sort(key=lambda x: x[1], reverse=True)
        return areas[:3]  # Return top 3 areas
    
    def _evaluate_blocking(
        self, 
        my_pos: Tuple[int, int], 
        opp_pos: Tuple[int, int],
        board: List[List[int]],
        state: dict
    ) -> float:
        """
        Evaluate how well this move blocks the opponent.
        
        Args:
            my_pos: Our new position
            opp_pos: Opponent position
            board: Board state
            state: Full game state
        
        Returns:
            Blocking score
        """
        # Predict opponent's next move
        opp_moves = self._predict_opponent_moves(opp_pos, board, state)
        
        if not opp_moves:
            return 0.0
        
        # Check if our move reduces opponent's options
        blocking_score = 0.0
        
        # Calculate opponent's accessible space after our move
        opp_accessible = self._calculate_accessible_space(opp_pos, board)
        
        # If we're cutting off a path, that's good
        if opp_accessible < 50:  # Threshold for "trapped"
            blocking_score += 5.0
        
        # Check if we're between opponent and large open areas
        open_areas = self._find_open_areas(board)
        for area_center, area_size in open_areas:
            if area_size > 20:  # Large area
                # Check if we're on the path between opponent and area
                if self._is_on_path(opp_pos, area_center, my_pos):
                    blocking_score += 3.0
        
        return blocking_score
    
    def _predict_opponent_moves(
        self, 
        opp_pos: Tuple[int, int], 
        board: List[List[int]],
        state: dict
    ) -> List[str]:
        """
        Predict opponent's likely next moves.
        
        Args:
            opp_pos: Opponent position
            board: Board state
            state: Full game state
        
        Returns:
            List of predicted direction names
        """
        valid_opp_moves = self._get_valid_moves(opp_pos, board)
        
        if not valid_opp_moves:
            return []
        
        # Simple prediction: opponent likely moves toward open space
        predicted = []
        
        for move in valid_opp_moves:
            dx, dy = self.directions[move]
            new_pos = (opp_pos[0] + dx, opp_pos[1] + dy)
            space = self._calculate_accessible_space(new_pos, board)
            
            if space > 30:  # Threshold
                predicted.append(move)
        
        return predicted if predicted else valid_opp_moves
    
    def _is_on_path(
        self, 
        start: Tuple[int, int], 
        end: Tuple[int, int], 
        point: Tuple[int, int]
    ) -> bool:
        """
        Check if a point is roughly on the path between start and end.
        
        Args:
            start: Start position
            end: End position
            point: Point to check
        
        Returns:
            True if point is on path
        """
        # Simple heuristic: check if point is within Manhattan distance
        path_dist = abs(start[0] - end[0]) + abs(start[1] - end[1])
        dist_via_point = (abs(start[0] - point[0]) + abs(start[1] - point[1]) + 
                         abs(point[0] - end[0]) + abs(point[1] - end[1]))
        
        return dist_via_point <= path_dist + 2
    
    def _is_corner(self, position: Tuple[int, int]) -> bool:
        """
        Check if position is in a corner.
        
        Args:
            position: Position to check
        
        Returns:
            True if in corner
        """
        x, y = position
        corners = [
            (0, 0), (self.board_width - 1, 0),
            (0, self.board_height - 1), 
            (self.board_width - 1, self.board_height - 1)
        ]
        return position in corners


# Global agent instance
agent = TronAgent()


def get_move(state: dict) -> str:
    """
    Main entry point for the game engine.
    
    This function is called by the game engine each tick.
    
    Args:
        state: Dictionary containing:
            - my_position: (x, y) tuple
            - opponent_position: (x, y) tuple
            - board: 2D list (0=empty, 1=wall/trail)
            - boosts: List of available boosts (optional)
            - turns_remaining: Number of turns left (optional)
            - opponent_last_direction: Last direction opponent moved (optional)
    
    Returns:
        str: Direction to move ('up', 'down', 'left', 'right')
    """
    return agent.get_move(state)



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
