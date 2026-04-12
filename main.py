import os
import discord
import requests
import json
from discord.ext import commands
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Discord bot setup
intents = discord.Intents.default()
intents.message_content = True  # Required to read message content
intents.messages = True
bot = commands.Bot(command_prefix="!", intents=intents)

# Store conversation history per channel/user for context
conversation_history = {}

# Hugging Face API configuration
HF_API_URL = "https://api-inference.huggingface.co/models/microsoft/Phi-3-mini-4k-instruct"
HF_HEADERS = {"Authorization": f"Bearer {os.environ.get('HF_TOKEN')}"}

async def get_ai_response(user_message, context_messages=None):
    """
    Get response from the Hugging Face model using requests
    """
    try:
        # Build conversation prompt
        prompt = build_conversation_prompt(user_message, context_messages)
        
        # Prepare the payload for the API
        payload = {
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": 500,
                "temperature": 0.7,
                "top_p": 0.95,
                "do_sample": True,
                "return_full_text": False
            }
        }
        
        # Make the API request
        response = requests.post(HF_API_URL, headers=HF_HEADERS, json=payload, timeout=30)
        
        # Check if request was successful
        if response.status_code == 200:
            result = response.json()
            # Extract the generated text
            if isinstance(result, list) and len(result) > 0:
                return result[0].get('generated_text', "Sorry, I couldn't generate a response.")
            elif isinstance(result, dict):
                return result.get('generated_text', "Sorry, I couldn't generate a response.")
            else:
                return str(result)
        else:
            print(f"API Error {response.status_code}: {response.text}")
            
            # Handle rate limiting
            if response.status_code == 429:
                return "I'm getting too many requests right now. Please try again in a few moments."
            # Handle model loading
            elif response.status_code == 503:
                return "The AI model is loading. Please try again in a few seconds."
            else:
                return f"Sorry, I'm having trouble thinking right now. (Error: {response.status_code})"
    
    except requests.exceptions.Timeout:
        print("Request timed out")
        return "The request timed out. Please try again."
    except requests.exceptions.ConnectionError:
        print("Connection error")
        return "I'm having trouble connecting to the AI service. Please check your internet connection."
    except Exception as e:
        print(f"Error getting AI response: {e}")
        return "Sorry, I'm having trouble thinking right now. Please try again later."

def build_conversation_prompt(user_message, context_messages=None):
    """
    Build a prompt for the model with conversation history
    """
    prompt = ""
    
    # Add system message
    prompt += "<|system|>\nYou are a helpful, friendly AI assistant on Discord. Keep responses concise but informative.\n"
    
    # Add conversation history if available
    if context_messages:
        for msg in context_messages[-10:]:  # Limit to last 10 messages for context
            if msg["role"] == "user":
                prompt += f"<|user|>\n{msg['content']}\n"
            elif msg["role"] == "assistant":
                prompt += f"<|assistant|>\n{msg['content']}\n"
    
    # Add current user message
    prompt += f"<|user|>\n{user_message}\n"
    prompt += "<|assistant|>\n"
    
    return prompt

def get_conversation_context(channel_id, user_id=None):
    """
    Retrieve conversation history for a channel or user
    """
    key = f"{channel_id}_{user_id}" if user_id else f"{channel_id}"
    return conversation_history.get(key, [])

def update_conversation_context(channel_id, user_message, bot_response, user_id=None):
    """
    Update conversation history
    """
    key = f"{channel_id}_{user_id}" if user_id else f"{channel_id}"
    
    if key not in conversation_history:
        conversation_history[key] = []
    
    # Add user message and bot response
    conversation_history[key].append({"role": "user", "content": user_message})
    conversation_history[key].append({"role": "assistant", "content": bot_response})
    
    # Keep only last 10 exchanges (20 messages total) to avoid context overflow
    if len(conversation_history[key]) > 20:
        conversation_history[key] = conversation_history[key][-20:]

@bot.event
async def on_ready():
    print(f'{bot.user} has connected to Discord!')
    print(f'Bot is in {len(bot.guilds)} guilds')
    
    # Set bot status
    await bot.change_presence(
        activity=discord.Activity(
            type=discord.ActivityType.listening,
            name="mentions and replies"
        )
    )

@bot.event
async def on_message(message):
    # Don't respond to the bot's own messages
    if message.author == bot.user:
        return
    
    # Check if bot was mentioned
    if bot.user in message.mentions:
        # Remove the mention from the message content
        content = message.content.replace(f'<@{bot.user.id}>', '').replace(f'<@!{bot.user.id}>', '').strip()
        
        # If message is just a mention without text, ask for input
        if not content:
            await message.reply("Hello! What would you like to talk about?")
            return
        
        # Send typing indicator
        async with message.channel.typing():
            # Get conversation context for this channel-user pair
            context = get_conversation_context(message.channel.id, message.author.id)
            
            # Get AI response
            response = await get_ai_response(content, context)
            
            # Update conversation context
            update_conversation_context(message.channel.id, content, response, message.author.id)
        
        # Reply to the message
        await message.reply(response)
    
    # Check if this is a reply to the bot's message
    elif message.reference and message.reference.resolved:
        referenced_msg = message.reference.resolved
        if referenced_msg.author == bot.user:
            # Get the content of the reply
            content = message.content.strip()
            
            if content:
                # Send typing indicator
                async with message.channel.typing():
                    # Get conversation context
                    context = get_conversation_context(message.channel.id, message.author.id)
                    
                    # Get AI response
                    response = await get_ai_response(content, context)
                    
                    # Update conversation context
                    update_conversation_context(message.channel.id, content, response, message.author.id)
                
                # Reply to the message
                await message.reply(response)
    
    # Process commands
    await bot.process_commands(message)

@bot.command(name='clear')
async def clear_context(ctx):
    """Clear conversation history for this channel"""
    key = f"{ctx.channel.id}_{ctx.author.id}"
    if key in conversation_history:
        del conversation_history[key]
    await ctx.send("✨ Conversation history cleared!")

@bot.command(name='info')
async def bot_info(ctx):
    """Show bot information"""
    embed = discord.Embed(
        title="AI Chat Bot Info",
        description="I'm an AI-powered Discord bot using Hugging Face's inference API!",
        color=discord.Color.blue()
    )
    embed.add_field(name="How to use me", value="• Mention me with @bot then your question\n• Reply to any of my messages", inline=False)
    embed.add_field(name="Features", value="• Remembers conversation context\n• Responds to mentions and replies\n• Per-user conversation history", inline=False)
    await ctx.send(embed=embed)

@bot.command(name='model')
async def change_model(ctx, model_name=None):
    """Change the AI model (owner only)"""
    if not model_name:
        await ctx.send("Current model: " + HF_API_URL.split('/')[-1])
        return
    
    # This would require restarting the bot or updating the global variable
    await ctx.send("Model changing is not fully implemented yet. Please restart the bot with a different model in the code.")

# Run the bot
if __name__ == "__main__":
    # Get Discord token from environment variable
    DISCORD_TOKEN = os.environ.get("DISCORD_TOKEN")
    
    if not DISCORD_TOKEN:
        print("Error: DISCORD_TOKEN not found in .env file")
        print("Please make sure your .env file contains: DISCORD_TOKEN=your_token_here")
    elif not os.environ.get("HF_TOKEN"):
        print("Warning: HF_TOKEN not found in .env file")
        print("The AI functionality won't work without it!")
        print("Please add: HF_TOKEN=your_huggingface_token_here")
    else:
        # Test the API connection
        print("Testing Hugging Face API connection...")
        test_payload = {
            "inputs": "Hello",
            "parameters": {"max_new_tokens": 10}
        }
        try:
            test_response = requests.post(HF_API_URL, headers=HF_HEADERS, json=test_payload, timeout=10)
            if test_response.status_code == 200:
                print("✓ API connection successful!")
            else:
                print(f"⚠ API test returned status {test_response.status_code}")
                print(f"  Response: {test_response.text[:200]}")
        except Exception as e:
            print(f"✗ API test failed: {e}")
        
        # Run the bot
        print("\nStarting Discord bot...")
        bot.run(DISCORD_TOKEN)
