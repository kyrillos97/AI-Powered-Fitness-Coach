import uuid
import json
from langgraph.graph import StateGraph, END
from typing import TypedDict
from logic.plan_generator import generate_initial_plan, generate_modified_plan
from logic.data_saver import save_plan_to_json


class AgentState(TypedDict):
    """
    Represents the state of our graph.
    It holds all data passed between nodes.
    """
    user_id: str
    user_data: dict
    diet_plan: dict
    workout_plan: dict
    user_input: str


def get_initial_user_data_node(state: AgentState):
    """Gathers initial user information."""
    print("🤖 Welcome! Let us create a personalized fitness plan for you.")
    height = input("📏 Enter your height in cm: ")
    weight = input("⚖️ Enter your weight in kg: ")
    age = input("🎂 Enter your age: ")
    gender = input("🚻 Enter your gender (male/female): ")
    goal = input("🎯 What is your goal? (For example: losing weight, building muscle) ")
    activity_level = input("🏃‍♂️ What is your activity level? (low/medium/high): ")
    
    user_data = {
        "height_cm": int(height),
        "weight_kg": int(weight),
        "age": int(age),
        "gender": gender,
        "goal": goal,
        "activity_level": activity_level
    }
    return {"user_data": user_data, "user_id": str(uuid.uuid4())}


def generate_plan_node(state: AgentState):
    """Generates the initial plan based on user data."""
    print("\n⏳ Creating your custom plan. This may take a few moments...")
    diet_plan, workout_plan = generate_initial_plan(state['user_data'])
    return {"diet_plan": diet_plan, "workout_plan": workout_plan}


def handle_user_feedback_node(state: AgentState):
    """Displays the plan and gets feedback from the user."""
    print("\n--- ✅ Your proposed plan✅ ---")
    print("\n🥗 Food plan:")
    print(json.dumps(state['diet_plan'], indent=2, ensure_ascii=False))
    print("\n🏋️‍♂️ Training plan:")
    print(json.dumps(state['workout_plan'], indent=2, ensure_ascii=False))
    
    user_input = input(
        "\n📝 Does the plan seem appropriate? (Type 'yes' to save, or request modifications such as 'reduce the number of exercises'): "
    )
    return {"user_input": user_input}


def modify_plan_node(state: AgentState):
    """Modifies the plan based on user feedback."""
    print("\n🔄 The plan is being modified based on your request...")
    new_diet_plan, new_workout_plan = generate_modified_plan(
        user_data=state['user_data'],
        old_diet_plan=state['diet_plan'],
        old_workout_plan=state['workout_plan'],
        modification_request=state['user_input']
    )
    return {"diet_plan": new_diet_plan, "workout_plan": new_workout_plan}


def save_plan_node(state: AgentState):
    """Saves the final approved plan."""
    print("\n💾 Saving your plan...")
    save_plan_to_json(state['user_id'], state['diet_plan'], state['workout_plan'])
    print("✅ Successfully saved! 🚀")
    return {}


def should_end_or_modify(state: AgentState):
    """
    This is a conditional edge. It decides where to go next.
    """
    if state['user_input'].strip().lower() == 'yes':
        return "save_plan"
    else:
        return "modify_plan"


# Build the LangGraph
builder = StateGraph(AgentState)

# Add nodes to the graph
builder.add_node("get_initial_data", get_initial_user_data_node)
builder.add_node("generate_plan", generate_plan_node)
builder.add_node("handle_feedback", handle_user_feedback_node)
builder.add_node("modify_plan", modify_plan_node)
builder.add_node("save_plan", save_plan_node)

# Set up the entry point and edges
builder.set_entry_point("get_initial_data")
builder.add_edge("get_initial_data", "generate_plan")
builder.add_edge("generate_plan", "handle_feedback")

# Define the conditional edge for the main loop
builder.add_conditional_edges("handle_feedback", should_end_or_modify, {
    "save_plan": "save_plan",
    "modify_plan": "modify_plan"
})

# Add the loop from modify_plan back to feedback
builder.add_edge("modify_plan", "handle_feedback")

# End the graph when the plan is saved
builder.add_edge("save_plan", END)

# Compile the graph
app = builder.compile()

if __name__ == "__main__":
    final_state = app.invoke({})
    