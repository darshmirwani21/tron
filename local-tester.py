import requests
import json
import time

# --- CONFIGURATION ---
# The URL of the agent to test. Assumes the participant is running their
# agent.py on the default port 5008.
AGENT_URL = "http://localhost:5008"

def run_test(name, test_function):
    """Helper function to run a test and print the outcome."""
    print(f"--- Running Test: {name} ---")
    try:
        success, message = test_function()
        if success:
            print(f"✅ PASS: {message}\n")
            return True
        else:
            print(f"❌ FAIL: {message}\n")
            return False
    except requests.exceptions.ConnectionError as e:
        print(f"❌ FAIL: Could not connect to the agent at {AGENT_URL}.")
        print("   Is your agent.py running in another terminal?\n")
        return False
    except Exception as e:
        print(f"❌ FAIL: An unexpected error occurred: {e}\n")
        return False

# --- API COMPLIANCE TESTS ---

def test_latency_check():
    """Tests the GET / endpoint."""
    response = requests.get(AGENT_URL, timeout=2)
    if response.status_code != 200:
        return False, f"Expected status code 200, but got {response.status_code}."
    
    try:
        data = response.json()
        if "participant" not in data or "agent_name" not in data:
            return False, "The JSON response is missing 'participant' or 'agent_name' keys."
        return True, f"Agent identified as {data['agent_name']} from {data['participant']}."
    except json.JSONDecodeError:
        return False, "The response was not valid JSON."

def test_send_state():
    """Tests the POST /send-state endpoint."""
    # Dummy state matching Case Closed game format
    dummy_state = {
        "board": [[0 for _ in range(20)] for _ in range(18)],
        "agent1_trail": [(1, 2), (2, 2)],
        "agent2_trail": [(17, 15), (16, 15)],
        "agent1_length": 2,
        "agent2_length": 2,
        "agent1_alive": True,
        "agent2_alive": True,
        "agent1_boosts": 3,
        "agent2_boosts": 3,
        "turn_count": 1,
        "player_number": 1,
    }
    response = requests.post(f"{AGENT_URL}/send-state", json=dummy_state, timeout=2)
    
    if response.status_code != 200:
        return False, f"Expected status code 200, but got {response.status_code}."
    
    return True, "Agent correctly acknowledged the game state POST."

def test_get_move():
    """Tests the GET /send-move endpoint."""
    # Test with query parameters as judge engine sends them
    params = {"player_number": 1, "attempt_number": 1, "random_moves_left": 5, "turn_count": 1}
    response = requests.get(f"{AGENT_URL}/send-move", params=params, timeout=2)
    
    if response.status_code != 200:
        return False, f"Expected status code 200, but got {response.status_code}."
        
    try:
        data = response.json()
        if "move" not in data:
            return False, "The JSON response is missing the 'move' key."
        
        move = data["move"].upper()
        
        # Valid moves can be "DIRECTION" or "DIRECTION:BOOST"
        if ":" in move:
            parts = move.split(":")
            if len(parts) != 2:
                return False, f"Invalid move format: {data['move']}. Expected 'DIRECTION' or 'DIRECTION:BOOST'."
            direction, modifier = parts
            if modifier != "BOOST":
                return False, f"Invalid modifier: {modifier}. Expected 'BOOST'."
        else:
            direction = move
        
        valid_directions = ["UP", "DOWN", "LEFT", "RIGHT"]
        if direction not in valid_directions:
            return False, f"The direction '{direction}' is not one of the valid directions: {valid_directions}."
            
        return True, f"Agent responded with a valid move: {data['move']}."
    except json.JSONDecodeError:
        return False, "The response was not valid JSON."
        
def test_get_move_with_boost():
    """Tests that the agent can return a boost move."""
    params = {"player_number": 1, "attempt_number": 1, "random_moves_left": 5, "turn_count": 50}
    response = requests.get(f"{AGENT_URL}/send-move", params=params, timeout=2)
    
    if response.status_code != 200:
        return False, f"Expected status code 200, but got {response.status_code}."
        
    try:
        data = response.json()
        move = data["move"].upper()
        
        # We're just checking that boost format is understood (not requiring it be used)
        if ":" in move:
            parts = move.split(":")
            if len(parts) == 2 and parts[1] == "BOOST":
                return True, f"Agent can return boost moves: {data['move']}."
        
        return True, f"Agent returned a valid move (boost format is optional): {data['move']}."
    except json.JSONDecodeError:
        return False, "The response was not valid JSON."

def test_end_game():
    """Tests the POST /end endpoint."""
    dummy_end_state = {
        "board": [[0 for _ in range(20)] for _ in range(18)],
        "agent1_trail": [(1, 2), (2, 2), (3, 2)],
        "agent2_trail": [(17, 15), (16, 15)],
        "agent1_length": 3,
        "agent2_length": 2,
        "agent1_alive": True,
        "agent2_alive": False,
        "agent1_boosts": 2,
        "agent2_boosts": 3,
        "turn_count": 100,
        "result": "AGENT1_WIN"
    }
    response = requests.post(f"{AGENT_URL}/end", json=dummy_end_state, timeout=2)
    
    if response.status_code != 200:
        return False, f"Expected status code 200, but got {response.status_code}."
    
    return True, "Agent correctly acknowledged the game end POST."

# --- MAIN EXECUTION ---
if __name__ == "__main__":
    print("🚀 Starting API Compliance Test for Case Closed Agent 🚀\n")
    
    # Give the agent a moment to start up
    time.sleep(1)
    
    results = [
        run_test("Initial Latency Check (GET /)", test_latency_check),
        run_test("Receive Game State (POST /send-state)", test_send_state),
        run_test("Send Move (GET /send-move)", test_get_move),
        run_test("Send Move with Boost Format", test_get_move_with_boost),
        run_test("End Game Notification (POST /end)", test_end_game),
    ]
    
    print("--- Test Summary ---")
    if all(results):
        print("🎉 CONGRATULATIONS! Your agent passed all API compliance checks! 🎉")
        print("You are ready to submit.")
    else:
        print("🚨 Your agent failed one or more compliance checks.")
        print("Please review the errors above, fix your agent.py, and run the tester again.")
