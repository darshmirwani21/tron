import random
from collections import deque
from enum import Enum
from typing import Optional

EMPTY = 0
AGENT = 1

"""
GameBoard class manages the game board.

Handles the 2D grid, state of each cell, and provides torus (wraparound)
functionality for all coordinate-based operations.
"""
class GameBoard:
    def __init__(self, height: int = 18, width: int = 20):
        self.height = height
        self.width = width
        self.grid = [[EMPTY for _ in range(width)] for _ in range(height)]

    def _torus_check(self, position: tuple[int, int]) -> tuple[int, int]:
        x, y = position
        normalized_x = x % self.width
        normalized_y = y % self.height
        return (normalized_x, normalized_y)
    
    def get_cell_state(self, position: tuple[int, int]) -> int:
        x, y = self._torus_check(position)
        return self.grid[y][x]

    def set_cell_state(self, position: tuple[int, int], state: int):
        x, y = self._torus_check(position)
        self.grid[y][x] = state

    def get_random_empty_cell(self) -> tuple[int, int] | None:
        empty_cells = []
        for y in range(self.height):
            for x in range(self.width):
                if self.grid[y][x] == EMPTY:
                    empty_cells.append((x, y))
        
        if not empty_cells:
            return None
        
        return random.choice(empty_cells)

    def __str__(self) -> str:
        chars = {EMPTY: '.', AGENT: 'A'}
        board_str = ""
        for y in range(self.height):
            for x in range(self.width):
                board_str += chars.get(self.grid[y][x], '?') + ' '
            board_str += '\n'
        return board_str


UP = (0, -1)
DOWN = (0, 1)
RIGHT = (1, 0)
LEFT = (-1, 0)

class Direction(Enum):
    UP = (0, -1)
    DOWN = (0, 1)
    RIGHT = (1, 0)
    LEFT = (-1, 0)

class GameResult(Enum):
    AGENT1_WIN = 1
    AGENT2_WIN = 2
    DRAW = 3

class Agent:
    '''This class represents an agent in the game. It manages the agent's trail using a deque.'''
    def __init__(self, agent_id: str, start_pos: tuple[int, int], start_dir: Direction, board: GameBoard):
        self.agent_id = agent_id
        second = (start_pos[0] + start_dir.value[0], start_pos[1] + start_dir.value[1])
        self.trail = deque([start_pos, second])  # Trail of positions
        self.direction = start_dir
        self.board = board
        self.alive = True
        self.length = 2  # Initial length of the trail
        self.boosts_remaining = 3  # Each agent gets 3 speed boosts

        self.board.set_cell_state(start_pos, AGENT)
        self.board.set_cell_state(second, AGENT)
    
    def is_head(self, position: tuple[int, int]) -> bool:
        return position == self.trail[-1]
    
    def move(self, direction: Direction, other_agent: Optional['Agent'] = None, use_boost: bool = False) -> bool:
        """
        Moves the agent in the given direction and handles collisions.
        Agents leave a permanent trail behind them.
        
        Args:
            direction: Direction enum indicating where to move
            other_agent: The other agent on the board (for collision detection)
            use_boost: If True and boosts available, moves twice instead of once
        
        Returns:
            True if the agent survives the move, False if it dies
        """
        if not self.alive:
            return False

        if use_boost and self.boosts_remaining <= 0:
            print(f'Agent {self.agent_id} tried to boost but has no boosts remaining')
            use_boost = False
        
        num_moves = 2 if use_boost else 1
        
        if use_boost:
            self.boosts_remaining -= 1
            print(f'Agent {self.agent_id} used boost! ({self.boosts_remaining} remaining)')
        
        for move_num in range(num_moves):
            cur_dx, cur_dy = self.direction.value
            req_dx, req_dy = direction.value
            if (req_dx, req_dy) == (-cur_dx, -cur_dy):
                print('invalid move')
                continue  # Skip this move if invalid direction
            
            head = self.trail[-1]
            dx, dy = direction.value
            new_head = (head[0] + dx, head[1] + dy)
            
            new_head = self.board._torus_check(new_head)
            
            cell_state = self.board.get_cell_state(new_head)
            
            self.direction = direction
            
            # Handle collision with agent trail
            if cell_state == AGENT:
                # Check if it's our own trail (any part of our trail)
                if new_head in self.trail:
                    # Hit our own trail
                    self.alive = False
                    return False
                
                # Check collision with the other agent
                if other_agent and other_agent.alive and new_head in other_agent.trail:
                    # Check for head-on collision
                    if other_agent.is_head(new_head):
                        # Head-on collision: always a draw (both agents die)
                        self.alive = False
                        other_agent.alive = False
                        return False
                    else:
                        # Hit other agent's trail (not head-on)
                        self.alive = False
                        return False
            
            # Normal move (empty cell) - leave trail behind
            # Add new head, trail keeps growing
            self.trail.append(new_head)
            self.length += 1
            self.board.set_cell_state(new_head, AGENT)
        
        return True

    def get_trail_positions(self) -> list[tuple[int, int]]:
        return list(self.trail)
    

class Game:
    def __init__(self):
        self.board = GameBoard()
        self.agent1 = Agent(agent_id=1, start_pos=(1, 2), start_dir=Direction.RIGHT, board=self.board)
        self.agent2 = Agent(agent_id=2, start_pos=(17, 15), start_dir=Direction.LEFT, board=self.board)
        self.turns = 0
    
    def reset(self):
        """Resets the game to the initial state."""
        self.board = GameBoard()
        self.agent1 = Agent(agent_id=1, start_pos=(1, 2), start_dir=Direction.RIGHT, board=self.board)
        self.agent2 = Agent(agent_id=2, start_pos=(17, 15), start_dir=Direction.LEFT, board=self.board)
        self.turns = 0
    
    def step(self, dir1: Direction, dir2: Direction, boost1: bool = False, boost2: bool = False):
        """Advances the game by one step, moving both agents."""
        if self.turns >= 200:
            print("Max turns reached. Checking trail lengths...")
            if self.agent1.length > self.agent2.length:
                print(f"Agent 1 wins with trail length {self.agent1.length} vs {self.agent2.length}")
                return GameResult.AGENT1_WIN
            elif self.agent2.length > self.agent1.length:
                print(f"Agent 2 wins with trail length {self.agent2.length} vs {self.agent1.length}")
                return GameResult.AGENT2_WIN
            else:
                print(f"Draw - both agents have trail length {self.agent1.length}")
                return GameResult.DRAW
        
        agent_one_alive = self.agent1.move(dir1, other_agent=self.agent2, use_boost=boost1)
        agent_two_alive = self.agent2.move(dir2, other_agent=self.agent1, use_boost=boost2)

        if not agent_one_alive and not agent_two_alive:
            print("Both agents have crashed.")
            return GameResult.DRAW
        elif not agent_one_alive:
            print("Agent 1 has crashed.")
            return GameResult.AGENT2_WIN
        elif not agent_two_alive:
            print("Agent 2 has crashed.")
            return GameResult.AGENT1_WIN

        self.turns += 1
