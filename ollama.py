
import requests
import os

# nsfw checker
# mesh checker

OLLAMA_URL = "http://localhost:11500" 
MODEL = "qwen2.5vl:7b"

# testar com modificadores de qualidade

def refine_prompt(user_prompt):
    payload ={
        "model": MODEL,
        "prompt": f"""
            You are a prompt engineer for text-to-3D/multiview generation.
            Goal: rewrite a messy input into a clean, structured description that keeps multi-word phrases intact.
            Rules:
            - Start with the object name (singular, lowercase).
            - Use concise, visual attributes.
            - Keep compound phrases together (examples: "lake in the middle", "red door on the left", "snow-capped mountain").
            - Preserve spatial relations ("at the center", "on the left/right", "surrounded by").
            - Do NOT split a phrase into separate tokens like "lake, middle". Keep it as "central lake" or "lake in the middle".
            - Prefer synonyms that compress relations: "lake in the middle" to "central lake".
            - Output ONLY the refined prompt.
            - No extra text.

            Examples:
            Input: "a cute orange cat"
            Output: "cat, cute, orange, detailed fur"

            Input: "a house with a red door on the left and a big tree behind"
            Output: "house, red door on the left, big tree behind"

            Input: "a city square with a fountain in the center and benches around"
            Output: "city square, central fountain, benches around"

            Input: "a big park with trails and trees and a big lake in the middle"
            Output: "park, big, trails, trees, central lake"

            Refine this prompt: {user_prompt}
            """,
        "max_tokens": 256,
        "temperature": 0.7,
        "top_p": 0.9,
        "n": 1,
        "stop": None,
        "stream": False
        
    }
    resp = requests.post(f"{OLLAMA_URL}/api/generate", json=payload)
    resp.raise_for_status()
    refined_prompt = resp.json()['response']
    return refined_prompt

def negative_prompt(user_prompt):
    payload ={
        "model": MODEL,
        "prompt": f"""
            You are a prompt engineer for text-to-3D/multiview generation.
            Goal: given a positive prompt, generate a concise negative prompt that lists unwanted attributes to avoid in the generated output.
            Rules:
            - Use concise, visual attributes.
            - 8-12 items max.
            - Focus on common issues in 3D generation: "blurry", "low detail", "bad anatomy"; but related to the object.
            - Be context-aware. Also focus on the prompt object and give relevant negative attributes related to it. Example: for "cat", include "extra limbs", "extra tail", "deformed face".
                * Humans/animals: anatomy terms allowed (example: extra limbs, deformed face).
                * Landscapes/scenes/objects: DO NOT use anatomy words or “incorrect <object>” phrases. Prefer concrete artifact terms (topology, textures, composition).
            - Prefer general artifact terms + a few scene-specific constraints. No redundancy.
            - Avoid overly specific or rare terms.
            - Output ONLY the negative prompt and on one line with commas separating attributes.
            - No extra text.

            Input: park, big, trails, trees, central lake
            Output: low detail, artifacts, duplicate trees, intersecting trails, overcrowded foliage, floating objects, geometry holes, texture stretching, tiling textures, flat water, blocky water surface, unrealistic water reflections, incorrect scale

            Generate the negative prompt: {user_prompt}
            """,
        "max_tokens": 256,
        "temperature": 0.7,
        "top_p": 0.9,
        "n": 1,
        "stop": None,
        "stream": False

    }
    resp = requests.post(f"{OLLAMA_URL}/api/generate", json=payload)
    resp.raise_for_status()
    refined_prompt = resp.json()['response']
    return refined_prompt


if __name__ == "__main__":
    print(refine_prompt("a big park with trails and trees and big lake in the middle"))
    print(negative_prompt("park, big, trails, trees, central lake"))



# old prompts:
"""
    You are a prompt engineer for text-to-3D mesh generation.
    Your task is to rewrite a user vague or messy input into a clean, structured positive prompt that only describes the object itself.
    You need to: always start with the object name, separate descriptors with commas.
    Only output the refined prompt, do not include any explanations or additional text.
    Here are some examples:
    User input: "a cute orange cat"
    Refined prompt: "cat, cute, orange, detailed fur"
    Refine this prompt: "{user_prompt}
"""