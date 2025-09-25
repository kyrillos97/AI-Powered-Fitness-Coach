from langchain_core.prompts import ChatPromptTemplate
from langchain.output_parsers import PydanticOutputParser
from models.plan_models import WorkoutPlan, DietPlan
import json
import os
from unittest.mock import Mock

# OpenAI + OpenRouter
from langchain_openai import ChatOpenAI


# Wrapper for OpenRouter (so you can keep using ChatOpenRouter in your code)
class ChatOpenRouter(ChatOpenAI):
    def __init__(self, model_name: str, temperature: float = 0, **kwargs):
        super().__init__(
            model=model_name,
            temperature=temperature,
            api_key=os.getenv("OPENROUTER_API_KEY"),
            base_url="https://openrouter.ai/api/v1",
            **kwargs
        )


def safe_parse(parser, raw_response, plan_type: str, retries: int = 3):
    """Try parsing JSON safely with retries, always returning a Pydantic model."""
    import re
    
    for attempt in range(1, retries + 1):
        try:
            raw_json = raw_response.content.strip()

            # ✅ إزالة ```json ... ``` من النص لو موجودة
            if raw_json.startswith("```"):
                raw_json = re.sub(r"^```[a-zA-Z]*\n", "", raw_json)
                raw_json = raw_json.rstrip("`").rstrip()
                if raw_json.endswith("```"):
                    raw_json = raw_json[:-3].strip()

            # ✅ parsing باستخدام الـ Pydantic parser → بيرجع model
            return parser.parse(raw_json)

        except Exception as e:
            print(f"❌ Invalid {plan_type} JSON, retrying ({attempt}/{retries})...")

    raise ValueError(
        f"Failed to parse {plan_type} after {retries} retries. Raw response:\n{raw_response.content}"
    )


def generate_initial_plan(user_data: dict):
    """Generates the initial diet and workout plan."""
    if not os.getenv("OPENROUTER_API_KEY"):
        llm = Mock()
        llm.invoke.return_value = Mock(content=json.dumps({
            "diet": [{"meal_time": "test", "items": ["test"], "calories": 1}],
            "workout": {"pushups": {"reps": [10, 12], "rest_minutes": 1.5}}
        }))
    else:
        llm = ChatOpenRouter(
            model_name="mistralai/mistral-7b-instruct:free",
            temperature=0,
            max_tokens=1600
        )
    
    diet_parser = PydanticOutputParser(pydantic_object=DietPlan)
    workout_parser = PydanticOutputParser(pydantic_object=WorkoutPlan)
    
    diet_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a professional nutritionist. 
Return STRICT JSON ONLY that matches the schema. 
No explanations, no markdown, no extra fields.
"""),
        ("human", """User Data: {user_data}

Create a **weekly diet plan** (Monday to Sunday). 

⚠️ RULES (must follow strictly):
- Each day must contain exactly 3 meals: breakfast, lunch, dinner, and optionally 1 snack.
- Each meal MUST include:
  - "meal_time" (string)
  - "items" (list of strings)
  - "calories" (integer, estimated calories for the meal)
- Do not omit "calories" for snacks or any meal.

Return JSON following EXACTLY this schema:
{format_instructions}

⚠️ IMPORTANT:
- Return JSON only
- No extra text or markdown
""")
    ])
    
    workout_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a professional fitness coach. 
Return STRICT JSON ONLY that matches the schema. 
No explanations, no markdown, no extra fields.
"""),
        ("human", """User Data: {user_data}

Return JSON following EXACTLY this schema:
{format_instructions}

⚠️ IMPORTANT:
- Return JSON only
- No extra text or markdown
""")
    ])
    
    diet_plan_raw = (diet_prompt | llm).invoke(
        {
            "user_data": str(user_data),
            "format_instructions": diet_parser.get_format_instructions()
        },
        config={"max_tokens": 1600}
    )
    workout_plan_raw = (workout_prompt | llm).invoke(
        {
            "user_data": str(user_data),
            "format_instructions": workout_parser.get_format_instructions()
        },
        config={"max_tokens": 1600}
    )
    
    parsed_diet_plan = safe_parse(diet_parser, diet_plan_raw, "diet plan")
    parsed_workout_plan = safe_parse(workout_parser, workout_plan_raw, "workout plan")
    
    return parsed_diet_plan.dict(), parsed_workout_plan.dict()


def generate_modified_plan(user_data, old_diet_plan, old_workout_plan, modification_request):
    """Generates a new plan based on an old plan and a user's modification request."""
    if not os.getenv("OPENROUTER_API_KEY"):
        llm = Mock()
        llm.invoke.return_value = Mock(content=json.dumps({
            "diet": [{"meal_time": "test", "items": ["test"], "calories": 1}],
            "workout": {"pushups": {"reps": [10, 12], "rest_minutes": 1.5}}
        }))
    else:
        llm = ChatOpenRouter(
            model_name="mistralai/mistral-7b-instruct:free",
            temperature=0,
            max_tokens=1600
        )
    
    diet_parser = PydanticOutputParser(pydantic_object=DietPlan)
    workout_parser = PydanticOutputParser(pydantic_object=WorkoutPlan)

    diet_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a professional nutritionist. 
Return STRICT JSON ONLY that matches the schema. 
No explanations, no markdown, no extra fields.
"""),
        ("human", """User Data: {user_data}

Current Diet Plan: {old_diet_plan}

Modification Request: {modification_request}

Update the plan into a **weekly diet plan** (Monday to Sunday). 

⚠️ RULES (must follow strictly):
- Each day must contain exactly 3 meals: breakfast, lunch, dinner, and optionally 1 snack.
- Each meal MUST include:
  - "meal_time" (string)
  - "items" (list of strings)
  - "calories" (integer, estimated calories for the meal)
- Do not omit "calories" for snacks or any meal.

Return JSON following EXACTLY this schema:
{format_instructions}

⚠️ IMPORTANT:
- Return JSON only
- No extra text or markdown
""")
    ])
    
    workout_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a professional fitness coach. 
Return STRICT JSON ONLY that matches the schema. 
No explanations, no markdown, no extra fields.
"""),
        ("human", """User Data: {user_data}

Current Workout Plan: {old_workout_plan}

Modification Request: {modification_request}

Return JSON following EXACTLY this schema:
{format_instructions}

⚠️ IMPORTANT:
- Return JSON only
- No extra text or markdown
""")
    ])

    new_diet_plan_raw = (diet_prompt | llm).invoke(
        {
            "user_data": str(user_data), 
            "old_diet_plan": json.dumps(old_diet_plan, ensure_ascii=False),
            "modification_request": modification_request,
            "format_instructions": diet_parser.get_format_instructions()
        },
        config={"max_tokens": 1600}
    )
    
    new_workout_plan_raw = (workout_prompt | llm).invoke(
        {
            "user_data": str(user_data), 
            "old_workout_plan": json.dumps(old_workout_plan, ensure_ascii=False),
            "modification_request": modification_request,
            "format_instructions": workout_parser.get_format_instructions()
        },
        config={"max_tokens": 1600}
    )
    
    parsed_new_diet_plan = safe_parse(diet_parser, new_diet_plan_raw, "new diet plan")
    parsed_new_workout_plan = safe_parse(workout_parser, new_workout_plan_raw, "new workout plan")
    
    return parsed_new_diet_plan.dict(), parsed_new_workout_plan.dict()
