import base64
import glob
import requests
import os
import json
from loguru import logger

logger.add("ollama.log", rotation="10 MB")

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://127.0.0.1:11200")
MODEL_TEXT = os.getenv("TEXT_MODEL", "qwen2.5:7b")
MODEL_VISION = os.getenv("VISION_MODEL", "qwen2.5vl:7b")

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
    
    
    prompt_text = f"""
        You are a prompt formatter for text-to-3D generation.
        YOU HAVE TWO TASKS.

        [TASK 1: REFINED POSITIVE PROMPT]
        GOAL: Convert user input into a comma-separated list of keywords AND descriptive phrases.

        CRITICAL RULES:
        1.  **PRESERVE PHRASES:** Do NOT split descriptive phrases. Keep attributes, actions, and locations attached to their subjects.
            * "hat with feathers" MUST remain "hat with feathers".
            * "riding a skateboard" MUST remain "riding a skateboard".
            * "with red wheels" MUST remain "with red wheels".
        2.  **PRESERVE ALL INFO:** You must include every detail from the input.
        3.  **ADD VIEW:**
            * If the subject is a person, animal, or creature, add ", full body".
            * If the subject is an object, building, or scene, add ", full view".
        4.  **OUTPUT:** Return ONLY the comma-separated list.

        EXAMPLES (Study these carefully):
        Input: "a cute orange cat"
        Output: "cat, cute, orange, full body"
        
        Input: "big red dog riding a skateboard"
        Output: "dog, big, red, riding a skateboard, full body"
        
        Input: "dinosaur wearing a hat with feathers and a trench coat"
        Output: "dinosaur, wearing a hat with feathers, wearing a trench coat, full body"
        
        Input: "purple ferrari with red wheels and a hat"
        Output: "ferrari, purple, with red wheels, with a hat, full view"

        Input: "golden knight holding a large sword, riding a black horse, the horse has wings and green eyes"
        Output: "golden knight, holding a large sword, riding a black horse, black horse with wings, black horse with green eyes, full body"

        [TASK 2: NEGATIVE PROMPT]
        Goal: Generate a standard negative prompt.
        
        MANDATORY ITEMS (always include):
        - nsfw, explicit content (ALWAYS FIRST)
        - blurry, low detail, artifacts
        - shadows, lighting, background (ALWAYS LAST)
        
        CONTEXT-AWARE (add 2-3 based on subject):
        - Humans/animals: bad anatomy, extra limbs, deformed face, missing limbs, deformed tail
        - Objects/Buildings: bad proportions, deformed parts, floating objects
        - Landscapes: (ignoring, as requested)
        
        Output: Comma-separated, one line.

        Examples:
        Input: "cat, cute, orange, full body"
        Output: "nsfw, explicit content, blurry, low detail, artifacts, bad anatomy, extra limbs, deformed face, deformed tail, shadows, lighting, background"

        Input: "dinosaur, wearing a hat with feathers, wearing a trench coat, full body"
        Output: "nsfw, explicit content, blurry, low detail, artifacts, bad anatomy, extra limbs, deformed face, deformed tail, shadows, lighting, background"

        Input: "ferrari, purple, with red wheels, with a hat, full view"
        Output: "nsfw, explicit content, blurry, low detail, artifacts, bad proportions, deformed parts, floating objects, shadows, lighting, background"

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
        "max_tokens": 300,
        "temperature": 0.1,  
        "top_p": 0.9,
        "stream": False,
        "keep_alive": "1h",
        "format": "json"
    }

    try:
        resp = requests.post(f"{OLLAMA_URL}/api/generate", json=payload, timeout=(10, 120))
        resp.raise_for_status()
        raw = resp.json()["response"]

        
        json_start = raw.find('{')
        json_end = raw.rfind('}') + 1
        if json_start == -1 or json_end == 0:
            logger.error(f"Resposta da API não continha JSON válido: {raw}")
            return {"refined": user_prompt, "negative": "nsfw, explicit content, blurry, low detail"}

        clean_raw = raw[json_start:json_end]
        data = json.loads(clean_raw)
        
    except requests.exceptions.RequestException as e:
        logger.error(f"Erro de request ao chamar Ollama: {e}")
        return {"refined": user_prompt, "negative": "nsfw, explicit content, blurry, low detail, artifacts"}
    except json.JSONDecodeError:
        logger.error(f"Falha ao decodificar JSON da resposta: {raw}")
        return {"refined": user_prompt, "negative": "nsfw, explicit content, blurry, low detail, artifacts"}

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
        
        CRITICAL ISSUES TO CHECK:
        - NSFW or explicit content
        - Geometry holes
        - Extra limbs (if the prompt doesn't call for it)
        - Missing body parts
        - Deformed face
        - Artifacts
        - Poor lighting/shadows
        
        If the cat has extra limbs, deformed face, an extra tail, say so.
        Return your answer strictly as a JSON object with two fields:

        {{
        "avaliation": "<describe if the images match the prompt, consistency, visible issues>",
        "negative_prompt": "<comma-separated short negative prompt with 6–10 concise unwanted attributes to avoid; if it is good, write 'none'; ALWAYS include 'nsfw, explicit content' first if any issues>"
        }}

        ONLY AVALIATION and NEGATIVE_PROMPT as keys.
        Positive prompt: {user_prompt}

        Output format example:
        {{
        "avaliation": "the cat has an extra tail and a deformed body. The images are not consistent and some views have an extra tail. the face has low detail. the colors are unrealistic and unmatched.",
        "negative_prompt": "nsfw, explicit content, blurry, low detail, artifacts, geometry holes, extra tail, deformed body"
        }}
        Return ONLY the JSON, no extra text.
        """.strip()

    payload = {
        "model": MODEL_VISION,
        "prompt": prompt_text,
        "images": [_b64(p) for p in pngs],
        "max_tokens": 300,
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
    
    prompts = prepare_prompts("big red dog, big tail, dark eyes, on a skate")
    print("Refined:", prompts['refined'])
    print("Negative:", prompts['negative'])
    print("\n---\n")
    
    prompts2 = prepare_prompts("a big park with trails and trees and big lake in the middle")
    print("Refined:", prompts2['refined'])
    print("Negative:", prompts2['negative'])



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