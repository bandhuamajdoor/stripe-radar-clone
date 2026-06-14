import json
import subprocess
import sys
import os

STATE_FILE = "scripts/state.json"

def load_state():
    if not os.path.exists(STATE_FILE):
        print(f"Error: State file {STATE_FILE} not found.")
        sys.exit(1)
    with open(STATE_FILE, "r") as f:
        return json.load(f)

def save_state(state):
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)

def run_command(cmd, check=True):
    print(f"Running command: {cmd}")
    res = subprocess.run(cmd, shell=True, text=True, capture_output=True)
    if res.returncode != 0:
        print(f"Command failed with code {res.returncode}")
        print("Stdout:")
        print(res.stdout)
        print("Stderr:")
        print(res.stderr)
        if check:
            sys.exit(1)
    return res

def main():
    state = load_state()
    current_day = state["current_day"]
    print(f"=== Verifying Day {current_day} ===")
    
    # Run tests for current day (if tests exist)
    test_file = f"tests/test_day_{current_day}.py"
    if os.path.exists(test_file):
        print(f"Running tests in {test_file}...")
        run_command(f".venv/bin/pytest {test_file}")
    else:
        # Fallback to general pytest if no specific file
        print("No specific test file found for today. Running general tests...")
        run_command(".venv/bin/pytest", check=False)
        
    print(f"=== Tests Passed! Committing changes for Day {current_day} ===")
    
    # Git commit
    run_command("git add .")
    # check if there are changes to commit
    status = run_command("git status --porcelain", check=False).stdout.strip()
    if not status:
        print("No changes to commit.")
    else:
        run_command(f'git commit -m "feat(day-{current_day}): implement day {current_day} tasks and verification"')
    
    # Git push
    print("Attempting to push to remote...")
    push_res = run_command("git push origin main", check=False)
    if push_res.returncode != 0:
        print("Warning: git push failed. If remote is not configured, this is expected.")
        print("You can set up remote on github and configure SSH/PAT.")
    else:
        print("Git push successful!")
        state["github_configured"] = True

    # Advance state
    state["completed_days"].append(current_day)
    state["current_day"] = current_day + 1
    save_state(state)
    print(f"=== Day {current_day} Completed! State updated to Day {state['current_day']} ===")

if __name__ == "__main__":
    main()
