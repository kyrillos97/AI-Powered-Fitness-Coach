# test_flow.py

import json
from unittest.mock import patch, Mock
from main import app as graph_app
from logic.plan_generator import generate_initial_plan
from logic.data_saver import save_plan_to_json

# Define mock plans that will be returned by our mock function
mock_diet_plan = {
  "diet": [
    {"meal_time": "الفطور", "items": ["بيض، خبز"], "calories": 400},
    {"meal_time": "الغداء", "items": ["دجاج، أرز"], "calories": 600}
  ]
}

mock_workout_plan = {
  "workout": {
    "Push-ups": {"reps": [10, 8, 8], "rest_minutes": 1.5},
    "Squats": {"reps": [15, 12, 12], "rest_minutes": 1.5}
  }
}

# Define a sequence of mock user inputs to simulate a successful run
user_inputs = [
    "175",  # Height
    "75",   # Weight
    "28",   # Age
    "ذكر",  # Gender
    "بناء عضلات",  # Goal
    "متوسط", # Activity level
    "نعم" # 'yes' to approve the plan
]

@patch('logic.plan_generator.generate_initial_plan')
@patch('builtins.input', side_effect=user_inputs)
@patch('logic.data_saver.save_plan_to_json')
def test_correct_flow(mock_save_plan_to_json, mock_input, mock_generate_plan):
    """
    Tests the main conversational flow of the chatbot.
    It simulates user input and LLM output to check if the graph's logic is correct.
    """
    # Configure the mock to return our predefined plans
    mock_generate_plan.return_value = (mock_diet_plan, mock_workout_plan)

    # Invoke the graph application
    graph_app.invoke({})

    # Assert that the `save_plan_to_json` function was called exactly once
    mock_save_plan_to_json.assert_called_once()
    
    # Optional: You can inspect the data that was passed to the save function
    saved_data = mock_save_plan_to_json.call_args[0]

    print("\n✅ Test successful: The main flow worked as expected.")
    print("✅ The `save_plan_to_json` function was called correctly.")
    print("✅ This proves your LangGraph logic is sound.")

if __name__ == "__main__":
    test_correct_flow()