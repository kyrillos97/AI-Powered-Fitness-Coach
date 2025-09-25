import requests
from langchain.tools import tool
from config import NUTRITIONIX_APP_ID, NUTRITIONIX_API_KEY
import json

@tool
def get_food_nutrition(food_name: str) -> str:
    """
    Useful for finding nutritional information about a food item. 
    The input should be a food name (e.g., '100g chicken breast').
    """
    url = "https://trackapi.nutritionix.com/v2/natural/nutrients"
    headers = {
        "Content-Type": "application/json",
        "x-app-id": NUTRITIONIX_APP_ID,
        "x-app-key": NUTRITIONIX_API_KEY
    }
    data = {"query": food_name}
    
    try:
        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()
        data = response.json()
        
        if not data.get('foods'):
            return "No nutritional information found for that food."
        
        food = data['foods'][0]
        result = {
            'food_name': food.get('food_name'),
            'calories': food.get('nf_calories'),
            'protein': food.get('nf_protein'),
            'carbs': food.get('nf_total_carbohydrate'),
            'fats': food.get('nf_total_fat')
        }
        return json.dumps(result, indent=2)
    except requests.exceptions.RequestException as e:
        return f"Error connecting to nutrition API: {e}"