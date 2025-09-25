#from pydantic import BaseModel, Field
#from typing import Dict, List, Any

#class ExerciseDetails(BaseModel):
#   reps: List[int] = Field(description="List of repetitions for each set.")
#    rest_minutes: float = Field(description="Rest duration between sets in minutes.")

#class WorkoutPlan(BaseModel):
#    workout: Dict[str, ExerciseDetails] = Field(description="Workout plan mapping exercise name to its details.")

#class MealDetails(BaseModel):
#    meal_time: str
#    items: List[str]
#    calories: int

#class DietPlan(BaseModel):
#    diet: List[MealDetails]





from pydantic import BaseModel, Field
from typing import Dict, List


# ---------------------------
# 🍽️ Diet Models
# ---------------------------

class MealDetails(BaseModel):
    meal_time: str = Field(
        description="The time at which a meal should be eaten (for example: breakfast, lunch, dinner)."
    )
    items: List[str] = Field(
        description="List of foods in the meal."
    )
    calories: int = Field(
        description="Approximate calories per meal."
    )


class DietPlan(BaseModel):
    diet: Dict[str, List[MealDetails]] = Field(
        description="Weekly diet plan, mapping each day of the week to a list of meals."
    )


# ---------------------------
# 🏋️ Workout Models
# ---------------------------

class ExerciseDetails(BaseModel):
    reps: List[int] = Field(
        description="List the number of repetitions for each set."
    )
    rest_minutes: float = Field(
        description="Rest time in minutes between sets."
    )


class DayPlan(BaseModel):
    exercises: Dict[str, ExerciseDetails] = Field(
        description="Exercises planned for the day, mapping exercise name to its details."
    )


class WorkoutPlan(BaseModel):
    workout: Dict[str, DayPlan] = Field(
        description="Weekly workout plan, mapping days to their exercises."
    )
