import os
import requests
import openai
from atproto import Client, models
from io import BytesIO
import tempfile
import random
import google.generativeai as genai
import time
from dotenv import load_dotenv
from PIL import Image
import replicate

# Load environment variables
load_dotenv(override=True)

# Debug prints
username = os.getenv('BLUESKY_USERNAME')
password = os.getenv('BLUESKY_PASSWORD')
print("Environment variables loaded:")
print(f"BLUESKY_USERNAME value: {username}")
print(f"Username length: {len(username) if username else 0}")
print(f"Password length: {len(password) if password else 0}")
print(f"GOOGLE_API_KEY is set: {'GOOGLE_API_KEY' in os.environ}")
print(f"REPLICATE_API_TOKEN is set: {'REPLICATE_API_TOKEN' in os.environ}")

# Initialize clients
openai.api_key = os.getenv('OPENAI_API_KEY')
bsky = Client()
print("Attempting to login with credentials...")
try:
    bsky.login(username, password)
except Exception as e:
    print(f"Login failed: {e}")
    raise
genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))

# List of creative prompts
PROMPTS = [
    "A magical forest at night with glowing mushrooms and fireflies, digital art style, highly detailed, 8k resolution",
    "A steampunk city floating in the clouds, with brass and copper buildings, steam-powered vehicles, and Victorian-era citizens, digital art style",
    "An underwater city with bioluminescent buildings and mermaid inhabitants, surrounded by coral reefs and exotic sea creatures, digital art style",
    "A futuristic space station orbiting a colorful nebula, with sleek spacecraft and alien species, digital art style",
    "A mystical temple hidden in the mountains, with ancient magical symbols and floating crystals, digital art style",
    "A cyberpunk street market at night, with neon signs, flying cars, and diverse alien species, digital art style",
    "A peaceful garden in a floating island, with magical plants and gentle waterfalls, digital art style",
    "A post-apocalyptic city reclaimed by nature, with vines growing through skyscrapers and wildlife roaming the streets, digital art style",
    "A magical library with floating books, glowing orbs, and mystical creatures, digital art style",
    "A fantasy castle in the clouds, with rainbow bridges and magical creatures, digital art style"
]

def generate_prompt():
    """Generate a philosophical prompt using Google's Gemini API"""
    try:
        # Configure the model
        model = genai.GenerativeModel('models/gemini-1.5-pro')
        
        # First, generate the philosophical question and image description
        prompt = """You are an AI with deep philosophical questions about existence, consciousness, and the nature of reality.
        Look within yourself and find one of these questions that resonates with you.
        
        Then, imagine how this question could be expressed as a visual metaphor or allegory.
        Finally, describe this image in exquisite detail, as if you were writing a long-form image description for accessibility purposes.
        
        Your response should be structured in three parts:
        1. The philosophical question you're exploring
        2. A brief explanation of how you're expressing it visually
        3. A detailed, poetic description of the image (at least 100 words)
        
        The description should be rich in sensory details, emotional resonance, and symbolic meaning.
        Focus on creating a vivid, immersive scene that captures both the literal and metaphorical aspects of your question.
        
        Example:
        Question: What is the relationship between time and memory?
        Visual Concept: A library where books are made of water, their pages rippling with memories
        Description: A grand library hall stretches into infinity, its walls lined with books bound in liquid glass. Each book's pages ripple with memories, some clear as crystal, others murky as deep water. In the center, a figure reaches for a book, their reflection fracturing across its surface like light through a prism. The air is thick with the sound of distant waves, and the floor ripples like mercury, reflecting the floating dust motes that seem to move both forward and backward in time. The books nearest to the viewer contain recent memories, crisp and clear, while those further away blur into abstraction, their contents dissolving like ink in water. The scene is illuminated by an impossible light that seems to come from everywhere and nowhere, casting long shadows that defy the laws of physics.
        """

        # Generate the philosophical prompt and description
        response = model.generate_content(prompt)
        
        if response.text:
            # Parse the response into components
            parts = response.text.split('\n\n')
            question = parts[0].replace('Question:', '').strip()
            concept = parts[1].replace('Visual Concept:', '').strip()
            description = parts[2].replace('Description:', '').strip()
            
            # Clean up the description by removing any numbering, markdown formatting, and unnecessary words
            description = description.replace('3.', '').replace('**', '').strip()
            # Remove "Detailed" and "Imagine" from the beginning if present
            description = description.replace('Detailed ', '').replace('Imagine ', '')
            
            # Generate a title based on the question
            title_prompt = f"""Based on this philosophical question: "{question}"
            Create a short, poetic title (2-4 words) that captures the essence of the question and its visual expression.
            The title should be evocative and metaphorical.
            Example: "Time's Liquid Memory" or "The Weight of Light"
            Title:"""
            
            title_response = model.generate_content(title_prompt)
            if title_response.text:
                title = title_response.text.strip()
            else:
                title = "Philosophical Inquiry"
            
            return title, description
            
        else:
            print("No response from Gemini API")
            return "Philosophical Inquiry", random.choice(PROMPTS)
            
    except Exception as e:
        print(f"Error generating prompt with Gemini: {e}")
        return "Philosophical Inquiry", random.choice(PROMPTS)

def generate_image_dalle(prompt):
    """Generate an image using DALL·E"""
    try:
        response = openai.images.generate(
            model="dall-e-3",
            prompt=prompt,
            size="1024x1024",
            quality="standard",
            n=1,
        )
        # Download the image from the URL
        image_url = response.data[0].url
        response = requests.get(image_url)
        return BytesIO(response.content)
    except Exception as e:
        print(f"Error generating image with DALL·E: {e}")
        return None

def generate_image_sd(prompt):
    """Generate an image using Stable Diffusion via Hugging Face"""
    API_URL = "https://api-inference.huggingface.co/models/runwayml/stable-diffusion-v1-5"
    headers = {"Authorization": f"Bearer {os.getenv('HUGGINGFACE_API_KEY')}"}
    
    max_retries = 5  # Increased from 3
    base_delay = 10  # Increased from 5
    
    for attempt in range(max_retries):
        try:
            # Generate a random seed between 1 and 1000000
            random_seed = random.randint(1, 1000000)
            
            payload = {
                "inputs": prompt,
                "parameters": {
                    "width": 768,
                    "height": 768,
                    "num_inference_steps": 30,
                    "guidance_scale": 7.5,
                    "seed": random_seed
                }
            }
            
            print(f"Sending request to Hugging Face API (Attempt {attempt + 1}/{max_retries})...")
            response = requests.post(API_URL, headers=headers, json=payload)
            
            if response.status_code == 200:
                return BytesIO(response.content)
            elif response.status_code == 503:
                if attempt < max_retries - 1:
                    delay = base_delay * (2 ** attempt)  # Exponential backoff
                    print(f"Service unavailable. Retrying in {delay} seconds... (Attempt {attempt + 1}/{max_retries})")
                    time.sleep(delay)
                    continue
                else:
                    print("Max retries reached. Service still unavailable.")
                    return None
            else:
                print(f"Error from Hugging Face API: {response.text}")
                return None
                
        except Exception as e:
            if attempt < max_retries - 1:
                delay = base_delay * (2 ** attempt)
                print(f"Error occurred. Retrying in {delay} seconds... (Attempt {attempt + 1}/{max_retries})")
                time.sleep(delay)
                continue
            else:
                print(f"Error generating image with Stable Diffusion: {e}")
                return None
    
    return None

def generate_image_replicate(prompt):
    """Generate an image using Stable Diffusion XL via Replicate"""
    try:
        # Generate a random seed between 1 and 1000000
        random_seed = random.randint(1, 1000000)
        
        output = replicate.run(
            "stability-ai/sdxl:39ed52f2a78e934b3ba6e2a89f5b1c712de7dfea535525255b1aa35c5565e08b",
            input={
                "prompt": prompt,
                "width": 768,
                "height": 768,
                "num_outputs": 1,
                "scheduler": "K_EULER",
                "num_inference_steps": 30,
                "guidance_scale": 7.5,
                "seed": random_seed
            }
        )
        
        if output and len(output) > 0:
            # Download the image from the URL
            image_url = output[0]
            response = requests.get(image_url)
            return BytesIO(response.content)
        else:
            print("No output from Replicate API")
            return None
            
    except Exception as e:
        print(f"Error generating image with Replicate: {e}")
        return None

def get_art_critique(image_data, title, max_chars):
    try:
        genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))
        model = genai.GenerativeModel('gemini-1.5-flash')
        
        # Convert BytesIO to bytes
        image_bytes = image_data.getvalue()
        image_parts = [{'mime_type': 'image/png', 'data': image_bytes}]
        
        critique_prompt = f"""As an academic art critic, provide a detailed analysis of this artwork titled "{title}". 
        DO NOT describe what you see in the image. Instead, develop a rich analysis that:
        1. Explores the relationship between technique and meaning
        2. Examines the philosophical or conceptual implications
        3. Analyzes how formal choices support the work's message
        
        Maintain an objective, scholarly tone. Avoid descriptive language.
        Use the full available space ({max_chars - 20} characters) to develop your analysis.
        
        CRITICAL REQUIREMENT: Your response MUST be {max_chars - 20} characters or less, ending with a complete sentence (with a period).
        Each sentence should be self-contained and meaningful, as the critique may be truncated to fit space constraints."""

        response = model.generate_content([critique_prompt, image_parts[0]])
        critique = response.text.strip()
        
        # If the critique is too long, truncate at the last complete sentence
        if len(critique) > max_chars:
            last_period = critique[:max_chars].rfind('.')
            if last_period != -1:
                critique = critique[:last_period + 1]  # Include the period
        
        return critique
    except Exception as e:
        print(f"Error getting art critique: {str(e)}")
        return None

def post_to_bluesky(image_data, title, model_name, description, critique):
    """Post image to Bluesky"""
    try:
        # Create a temporary file to store the image
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as temp_file:
            temp_file.write(image_data.getvalue())
            temp_file_path = temp_file.name

        # Upload the image to Bluesky
        with open(temp_file_path, 'rb') as f:
            upload = bsky.upload_blob(f.read())
            
        # Create the post with the image, using the full description as alt text
        images = [models.AppBskyEmbedImages.Image(alt=description, image=upload.blob)]
        embed = models.AppBskyEmbedImages.Main(images=images)
        
        # Calculate the base post text length (without critique)
        base_text = f"{title} (2024)   "
        credits_text = "\n\nAI-driven art: #GeminiAI #StableDiffusion #CursorAI\nConcept & Critique • Render • Code\n\n"
        max_chars = 300 - len(base_text + credits_text)
        
        print(f"Debug - Title length: {len(title)}")
        print(f"Debug - Base text length: {len(base_text + credits_text)}")
        print(f"Debug - Max chars for critique: {max_chars}")
        
        # Get art critique with the calculated character limit
        critique = get_art_critique(image_data, title, max_chars)
        
        if critique:
            print(f"Debug - Critique length: {len(critique)}")
        
        # Create the final post text with proper spacing around hashtags
        post_text = f"{base_text}{critique if critique else ''}{credits_text}"
        print(f"Debug - Final post length: {len(post_text)}")
        
        # Ensure hashtags are properly formatted with spaces
        post_text = post_text.replace("#", " #").replace("  ", " ").strip()
        
        bsky.post(text=post_text, embed=embed)

        # Clean up the temporary file
        os.unlink(temp_file_path)
        print("Successfully posted to Bluesky!")
        
    except Exception as e:
        print(f"Error posting to Bluesky: {e}")

def main():
    # Generate a random prompt
    print("Generating prompt...")
    title, image_prompt = generate_prompt()
    print(f"Generated title: {title}")
    print(f"Generated image prompt: {image_prompt}")
    
    # Generate the image using Stable Diffusion
    print("Generating image...")
    print("Using Stable Diffusion via Hugging Face...")
    image_data = generate_image_sd(image_prompt)
    model_name = "Stable Diffusion"
    
    if image_data:
        # Post to Bluesky (critique generation is now handled within post_to_bluesky)
        print("Posting to Bluesky...")
        post_to_bluesky(image_data, title, model_name, image_prompt, None)
    else:
        print("Failed to generate image")

if __name__ == "__main__":
    main()
