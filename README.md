# AI Art Post Bot

An automated bot that generates and posts AI art to Bluesky every hour. The bot uses:
- Google's Gemini API for generating philosophical prompts and critiques
- Stable Diffusion for image generation
- Bluesky API for posting

## Setup

1. Fork this repository
2. Add the following secrets to your repository:
   - `BLUESKY_USERNAME`: Your Bluesky username
   - `BLUESKY_PASSWORD`: Your Bluesky app-specific password
   - `GOOGLE_API_KEY`: Your Google API key for Gemini
   - `HUGGINGFACE_API_KEY`: Your Hugging Face API key

To add secrets:
1. Go to your repository settings
2. Click on "Secrets and variables" → "Actions"
3. Click "New repository secret"
4. Add each secret with its corresponding value

## How it Works

The bot runs every hour using GitHub Actions. For each run, it:
1. Generates a philosophical prompt using Gemini
2. Creates an image using Stable Diffusion
3. Generates an art critique using Gemini
4. Posts everything to Bluesky

## Manual Trigger

You can manually trigger the workflow by:
1. Going to the "Actions" tab
2. Selecting "Hourly AI Art Post"
3. Clicking "Run workflow"

## Local Development

To run the bot locally:
1. Clone the repository
2. Create a `.env` file with your API keys
3. Install dependencies: `pip install -r requirements.txt`
4. Run the script: `python ai_post_bot.py` 