import os
import json
import subprocess

def test_project_structure():
    # Assert directories exist
    dirs = ["src", "tests", "models", "data", "scripts"]
    for d in dirs:
        assert os.path.isdir(d), f"Directory {d} does not exist"

def test_requirements_file():
    assert os.path.isfile("requirements.txt"), "requirements.txt is missing"
    with open("requirements.txt", "r") as f:
        content = f.read()
        assert "fastapi" in content
        assert "scikit-learn" in content
        assert "confluent-kafka" in content

def test_venv_exists():
    assert os.path.isdir(".venv"), "Virtual environment .venv does not exist"
    assert os.path.isfile(".venv/bin/activate"), "Virtual environment activation script is missing"

def test_state_file():
    assert os.path.isfile("scripts/state.json"), "state.json is missing"
    with open("scripts/state.json", "r") as f:
        state = json.load(f)
        assert state["current_day"] == 1
        assert state["project_name"] == "stripe-radar-clone"

def test_git_initialized():
    assert os.path.isdir(".git"), ".git directory is missing"
    res = subprocess.run("git branch --show-current", shell=True, capture_output=True, text=True)
    assert res.returncode == 0
    assert res.stdout.strip() == "main"
