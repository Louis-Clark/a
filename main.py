import os
import discord
from discord.ext import commands
from dotenv import load_dotenv
from huggingface_hub import InferenceClient

# Load environment variables
load_dotenv()

# Discord bot setup
intents = discord.Intents.default()
intents.message_content = True
intents.messages = True
bot = commands.Bot(command_prefix="!", intents=intents)

# Store conversation history
conversation_history = {}

# Initialize the Hugging Face Inference Client
# This automatically uses the router and handles provider selection [citation:3][citation:5]
HF_TOKEN = os.environ.get("HF_TOKEN")
if HF_TOKEN:
    # The client automatically uses https://router.huggingface.co
    hf_client = InferenceClient(token=HF_TOKEN)
else:
    print("Warning: HF_TOKEN not found in environment variables")
    hf_client = None

# List of recommended models that work well with the router [citation:5]
AVAILABLE_MODELS = [
    "meta-llama/Llama-3.2-3B-Instruct",  # Fast, good for most tasks
    "HuggingFaceH4/zephyr-7b-beta",      # Great for conversations
    "microsoft/Phi-3-mini-4k-instruct",  # Small and efficient
    "Qwen/Qwen2.5-7B-Instruct",          # Strong performance
]

async def get_ai_response(user_message, context_messages=None, model=AVAILABLE_MODELS[0]):
    """
    Get response from Hugging Face model using the router
    """
    if not hf_client:
        return "AI service is not configured. Please check your HF_TOKEN."

    try:
        # Prepare messages for chat completion
        messages = []
        
        # Add system message
        messages.append({
            "role": "system",
            "content": "You are a helpful, friendly AI assistant on Discord. Keep responses concise and conversational."
        })
        
        # Add conversation history (last 6 exchanges for context)
        if context_messages:
            for msg in context_messages[-6:]:
                messages.append({
                    "role": msg["role"],
                    "content": msg["content"]
                })
        
        # Add current user message
        messages.append({
            "role": "user",
            "content": user_message
        })
        
        # Use the chat_completion method which works with the router [citation:9]
        response = hf_client.chat_completion(
            model=model,
            messages=messages,
            max_tokens=500,
            temperature=0.7,
            top_p=0.95
        )
        
        # Extract the response text
        if response and response.choices:
            return response.choices[0].message.content
        else:
            return "Sorry, I received an unexpected response format."

    except Exception as e:
        print(f"Error details: {type(e).__name__}: {e}")
        
        # Common error handling
        error_str = str(e).lower()
        if "rate limit" in error_str or "429" in error_str:
            return "I'm receiving too many requests. Please try again in a moment."
        elif "loading" in error_str or "503" in error_str:
            return "The AI model is waking up. Please try again in 5-10 seconds."
        elif "token" in error_str or "401" in error_str:
            return "There's an issue with my API authentication. Please contact the bot administrator."
        else:
            return f"Sorry, I encountered an error: {str(e)[:100]}"

def get_conversation_context(channel_id, user_id=None):
    """Retrieve conversation history"""
    key = f"{channel_id}_{user_id}" if user_id else f"{channel_id}"
    return conversation_history.get(key, [])

def update_conversation_context(channel_id, user_message, bot_response, user_id=None):
    """Update conversation history"""
    key = f"{channel_id}_{user_id}" if user_id else f"{channel_id}"
    
    if key not in conversation_history:
        conversation_history[key] = []
    
    conversation_history[key].append({"role": "user", "content": user_message})
    conversation_history[key].append({"role": "assistant", "content": bot_response})
    
    # Keep only last 10 exchanges
    if len(conversation_history[key]) > 20:
        conversation_history[key] = conversation_history[key][-20:]

@bot.event
async def on_ready():
    print(f'{bot.user} has connected to Discord!')
    print(f'Bot is in {len(bot.guilds)} guilds')
    
    # Test HF connection
    if hf_client:
        print("✓ Hugging Face client initialized successfully")
    else:
        print("✗ Hugging Face client failed to initialize")
    
    await bot.change_presence(
        activity=discord.Activity(
            type=discord.ActivityType.listening,
            name="mentions | !info"
        )
    )

@bot.event
async def on_message(message):
    if message.author == bot.user:
        return
    
    # Check for bot mention
    if bot.user in message.mentions:
        content = message.content.replace(f'<@{bot.user.id}>', '').replace(f'<@!{bot.user.id}>', '').strip()
        
        if not content:
            await message.reply("Hello! What would you like to talk about?")
            return
        
        async with message.channel.typing():
            context = get_conversation_context(message.channel.id, message.author.id)
            response = await get_ai_response(content, context)
            update_conversation_context(message.channel.id, content, response, message.author.id)
        
        await message.reply(response)
    
    # Check for reply to bot
    elif message.reference and message.reference.resolved:
        referenced_msg = message.reference.resolved
        if referenced_msg.author == bot.user:
            content = message.content.strip()
            if content:
                async with message.channel.typing():
                    context = get_conversation_context(message.channel.id, message.author.id)
                    response = await get_ai_response(content, context)
                    update_conversation_context(message.channel.id, content, response, message.author.id)
                await message.reply(response)
    
    await bot.process_commands(message)

@bot.command(name='clear')
async def clear_context(ctx):
    """Clear conversation history"""
    key = f"{ctx.channel.id}_{ctx.author.id}"
    if key in conversation_history:
        del conversation_history[key]
    await ctx.send("✨ Conversation history cleared!")

@bot.command(name='model')
async def switch_model(ctx, model_choice: str = None):
    """Switch between available AI models"""
    if not model_choice:
        model_list = "\n".join([f"• `{i}`: {model.split('/')[-1]}" for i, model in enumerate(AVAILABLE_MODELS)])
        await ctx.send(f"Available models:\n{model_list}\n\nUsage: `!model 0` to select the first model")
        return
    
    try:
        index = int(model_choice)
        if 0 <= index < len(AVAILABLE_MODELS):
            # Store the selected model for this user/channel
            key = f"model_{ctx.channel.id}_{ctx.author.id}"
            conversation_history[key] = AVAILABLE_MODELS[index]
            await ctx.send(f"✅ Switched to model: **{AVAILABLE_MODELS[index].split('/')[-1]}**")
        else:
            await ctx.send(f"Invalid model number. Choose 0-{len(AVAILABLE_MODELS)-1}")
    except ValueError:
        await ctx.send("Please provide a number. Use `!model` to see available models.")

@bot.command(name='info')
async def bot_info(ctx):
    """Show bot information"""
    embed = discord.Embed(
        title="AI Chat Bot",
        description="Powered by Hugging Face Inference Providers",
        color=discord.Color.blue()
    )
    embed.add_field(name="How to use", value="• Mention me with `@bot` then your question\n• Reply to any of my messages", inline=False)
    embed.add_field(name="Features", value="• Remembers conversation context\n• Multiple AI models available\n• Fast response times", inline=False)
    await ctx.send(embed=embed)

# Run the bot
if __name__ == "__main__":
    DISCORD_TOKEN = os.environ.get("DISCORD_TOKEN")
    
    if not DISCORD_TOKEN:
        print("Error: DISCORD_TOKEN not found in .env file")
    elif not HF_TOKEN:
        print("Error: HF_TOKEN not found in .env file")
        print("Get your token from: https://huggingface.co/settings/tokens")
    else:
        print("Starting Discord bot...")
        bot.run(DISCORD_TOKEN)
