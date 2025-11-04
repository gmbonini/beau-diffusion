import base64
import glob
import requests
import os
import json
from loguru import logger

logger.add("ollama.log", rotation="10 MB")

# nsfw checker
# mesh checker

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://127.0.0.1:11200")
MODEL_TEXT = os.getenv("TEXT_MODEL", "qwen2.5:7b")
MODEL_VISION = os.getenv("VISION_MODEL", "qwen2.5vl:7b")

# testar com modificadores de qualidade

def _b64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def start_ollama():
    """Check if Ollama is running and load a model"""
    url = "http://127.0.0.1:11200/api/tags"
    
    try:        
        r = requests.get(url, timeout=5)
        r.raise_for_status()
        
        models = r.json().get('models', [])
        if not models:
            print("Warning: No models available in Ollama")
            return
                    
        model_name = models[0]['name']  
        
        generate_url = "http://127.0.0.1:11200/api/generate"
        payload = {
            "model": model_name,
            "prompt": "test",
            "stream": False
        }
        
        r = requests.post(generate_url, json=payload, timeout=10)
        r.raise_for_status()
        
        print(f"✓ Ollama is running with model: {model_name}")
        
    except requests.exceptions.RequestException as e:
        print(f"Warning: Could not connect to Ollama: {e}")

def is_ollama_loaded(model=None):
    """Verifica se um modelo específico está carregado, ou ambos se None"""
    try:
        resp = requests.get(f"{OLLAMA_URL}/api/ps", timeout=5)
        data = resp.json()
        loaded_models = {m.get("model") for m in data.get("models", []) if m.get("loaded")}
        
        if model:
            return model in loaded_models
        
        return MODEL_TEXT in loaded_models
    except Exception:
        return False
    
def prepare_prompts(user_prompt: str) -> dict:
    """Usa modelo de TEXTO para processar prompts"""
    
    # --- PROMPT CORRIGIDO ABAIXO ---
    prompt_text = f"""
        You are a prompt engineer for text-to-3D/multiview generation.
        YOU HAVE TWO TASKS.

        [TASK 1: REFINED POSITIVE PROMPT]
        Goal: rewrite a messy input into a clean, structured description, PRESERVING all key attributes.
        Rules:
        - Start with the object name (singular, lowercase). Example: "red dog..." > start with "dog".
        - Preserve ALL important object attributes, styles, and features from the input.
        - Only ignore clear background/environment details (e.g., "in a park", "on the street"). Focus on the object itself.
        - Use concise, visual attributes, separated by commas.
        - Keep compound phrases together (examples: "red door on the left", "snow-capped mountain").
        - Add "full body" if the prompt is about an animal, human or any living creature, unless a specific part is mentioned (e.g., "head of a cat"). If its about an object, place, or landscape, add "full view".
        - Output ONLY the refined prompt.
        - No extra text.

        Examples:
        Input: "a cute orange cat"
        Output: "cat, cute, orange, detailed fur, full body"
        
        Input: "red dog 4k realistic, anime, horns, giant fluffy tail"
        Output: "dog, red, 4k realistic, anime, horns, giant fluffy tail, full body"

        Input: "a house with a red door on the left and a big tree behind"
        Output: "house, red door on the left, big tree behind"

        Input: "a big park with trails and trees and a big lake in the middle"
        Output: "park, big, trails, trees, central lake"

        [TASK 2: NEGATIVE PROMPT]
        Goal: given the REFINED prompt from TASK 1, generate a concise negative prompt.
        Rules:
        - Start with 3-4 general 3D generation issues (e.g., "blurry", "low detail", "artifacts", "geometry holes").
        - Add 2-3 context-aware attributes related to the *refined positive prompt*.
          * Humans/animals: "bad anatomy", "extra limbs", "deformed face".
          * Objects: "bad proportions", "deformed parts".
          * Landscapes: "flat water", "intersecting trails", "floating objects".
        - ALWAYS add "shadows", "lighting", "background" at the end (we want to control these separately).
        - Total list should be 7-10 items, comma-separated on one line.
        - Do NOT use "incorrect <object>" phrases.
        - Output ONLY the negative prompt.
        - No extra text.

        Examples:
        Input: "cat, cute, orange, detailed fur, full body"
        Output: "blurry, low detail, artifacts, bad anatomy, extra limbs, deformed face, shadows, lighting, background"

        Input: "park, big, trails, trees, central lake"
        Output: "blurry, low detail, artifacts, intersecting trails, floating objects, flat water, unrealistic water, shadows, lighting, background"

        Input: "dog, red, 4k realistic, anime, horns, giant fluffy tail, full body"
        Output: "blurry, low detail, artifacts, bad anatomy, extra limbs, deformed horns, deformed tail, shadows, lighting, background"


        RETURN FORMAT (MANDATORY):
        Return ONLY this JSON (no extra text):
        {{
        "refined": "<refined positive prompt from TASK 1>",
        "negative": "<negative prompt from TASK 2>"
        }}

        Input: {user_prompt}
        """.strip()

    logger.info(f"[OLLAMA] prepare_prompts: preparing prompts for input: {user_prompt} (using {MODEL_TEXT})")
    
    payload = {
        "model": MODEL_TEXT,
        "prompt": prompt_text,
        "max_tokens": 256,
        "temperature": 0.3,
        "top_p": 0.9,
        "stream": False,
        "keep_alive": "1h",
        "format": "json"
    }

    resp = requests.post(f"{OLLAMA_URL}/api/generate", json=payload, timeout=(10, 120))
    resp.raise_for_status()
    raw = resp.json()["response"]

    try:
        json_start = raw.find('{')
        json_end = raw.rfind('}') + 1
        if json_start == -1 or json_end == 0:
            logger.error(f"Resposta da API não continha JSON válido: {raw}")
            return {"refined": user_prompt, "negative": "blurry, low detail"} # Fallback

        clean_raw = raw[json_start:json_end]
        data = json.loads(clean_raw)
        
    except json.JSONDecodeError:
        logger.error(f"Falha ao decodificar JSON da resposta: {raw}")
        return {"refined": user_prompt, "negative": "blurry, low detail, artifacts"}

    print(data)
    logger.info(f"[MV-ADAPTER] prepare_prompts result: {data}")

    return data

def check_views(dir_path, user_prompt):
    pngs = sorted(glob.glob(os.path.join(dir_path, "*.png")))

    prompt_text = f"""
        You are an image quality reviewer for 3D multiview generation.

        Given the target prompt and the attached images, analyze what you see.
        You need to evaluate:
        1) How well do the images match the prompt? (accuracy to the prompt)
        2) Are the images consistent with each other? (consistency across the multi views. Remember, it's a SINGLE object from DIFFERENT angles)
        3) Identify any visible issues/artifacts in the images. (image quality)

        You can be critical but not too much in your evaluation. If the images have minor issues but overall look good, say so. If they have major problems, point them out clearly. Only give constructive feedback.
        If the overall quality is good, you don't need to list every small issue and the negative prompt can be "none".
        The bigger problems are: geometry holes, extra limbs (if the prompt doesn't call for it), deformed face, artifacts, lighting, shadows, background.
        If the cat has extra limbs, deformed face, a extra tail, say so.
        Return your answer strictly as a JSON object with two fields:

        {{
        "avaliation": "<describe if the images match the prompt, consistency, visible issues>",
        "negative_prompt": "<comma-separated short negative prompt with 6–8 concise unwanted attributes to avoid; if it is good, write 'none'>"
        }}

        ONLY AVALIATION and NEGATIVE_PROMPT as keys.
        Positive prompt: {user_prompt}

        Output format example:
        {{
        "avaliation": "the cat have a extra tail and a deformed body. The images are not consistent and some views have a extra tail. the face has low detail. the colors are unrealistic and unmatched.",
        "negative_prompt": "blurry, low detail, artifacts, geometry holes, extra tail"
        }}
        Return ONLY the JSON, no extra text.
        """.strip()

    payload = {
        "model": MODEL_VISION,
        "prompt": prompt_text,
        "images": [_b64(p) for p in pngs],
        "max_tokens": 256,
        "temperature": 0.3,
        "top_p": 0.9,
        "stream": False,
        "format": "json",
        "keep_alive": "1h"
    }
    
    logger.info(f"[MV-ADAPTER] check_views loading (using {MODEL_VISION})")
    resp = requests.post(f"{OLLAMA_URL}/api/generate", json=payload, timeout=180)
    resp.raise_for_status()
    
    raw = resp.json()["response"]
    
    data = json.loads(raw)
    logger.info(f"[MV-ADAPTER] check_views result: {data}")
    return data


if __name__ == "__main__":
    # print(refine_prompt("a big park with trails and trees and big lake in the middle"))
    # print(negative_prompt("park, big, trails, trees, central lake"))
    # describe_pngs_in_dir("/tmp/views_ibp6tnyi/")
    # result = check_views("/tmp/views_ibp6tnyi/", "a big fat cat with big tail")
    # print(result)
    # print(result['avaliation'])
    prompts = prepare_prompts("a big park with trails and trees and big lake in the middle")
    print(prompts['refined'])
    print(prompts['negative'])


# ----------------- TESTING
def check_image(image_path):
    """Usa modelo de VISÃO para análise de imagem única"""
    with open(image_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")

    payload = {
        "model": MODEL_VISION,
        "prompt": "Describe what you see in the image in one concise sentence.",
        "images": [b64],
        "max_tokens": 128,
        "temperature": 0.2,
        "top_p": 0.9,
        "stream": False,
        "keep_alive": "1h",
    }
    logger.info(f"[TESTING] check_image loading (using {MODEL_VISION})")
    resp = requests.post(f"{OLLAMA_URL}/api/generate", json=payload, timeout=120)
    resp.raise_for_status()
    return resp.json()["response"].strip()

def describe_pngs_in_dir(dir_path):
    """Usa modelo de VISÃO para descrever múltiplas imagens"""
    pngs = sorted(glob.glob(os.path.join(dir_path, "*.png")))
    results = []
    for p in pngs:
        desc = check_image(p)
        print(f"{os.path.basename(p)}: {desc}")
        results.append((p, desc))
    return results