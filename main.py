import os
import discord
from discord.ext import commands
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize OpenAI client for Hugging Face inference
client = OpenAI(
    base_url="https://router.huggingface.co/hf-inference/v1",
    api_key=os.environ.get("HF_TOKEN"),
)

# Discord bot setup
intents = discord.Intents.default()
intents.message_content = True  # Required to read message content
intents.messages = True
bot = commands.Bot(command_prefix="!", intents=intents)

# Store conversation history per channel/user for context (optional)
conversation_history = {}

async def get_ai_response(user_message, context_messages=None):
    """
    Get response from the Hugging Face model
    """
    try:
        # Prepare messages array for the API
        messages = []
        
        # Add conversation context if available (last 5 exchanges)
        if context_messages:
            messages.extend(context_messages)
        
        # Add the current user message
        messages.append({
            "role": "user",
            "content": user_message
        })
        
        # Make the API call
        completion = client.chat.completions.create(
            model="cognitivecomputations/dolphin-2.6-mixtral-8x7b",
            messages=messages,
            max_tokens=500,  # Limit response length
            temperature=0.7  # Slight randomness for natural responses
        )
        
        return completion.choices[0].message.content
    
    except Exception as e:
        print(f"Error getting AI response: {e}")
        return "Sorry, I'm having trouble thinking right now. Please try again later."

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
        
        # Reply to the message (this will mention the user who asked)
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
    
    # Process commands (if any)
    await bot.process_commands(message)

# Optional: Command to clear conversation history for current channel/user
@bot.command(name='clear')
async def clear_context(ctx):
    """Clear conversation history for this channel"""
    key = f"{ctx.channel.id}_{ctx.author.id}"
    if key in conversation_history:
        del conversation_history[key]
    await ctx.send("✨ Conversation history cleared!")

# Optional: Command to get bot info
@bot.command(name='info')
async def bot_info(ctx):
    """Show bot information"""
    embed = discord.Embed(
        title="AI Chat Bot Info",
        description="I'm an AI-powered Discord bot that uses Hugging Face's inference API!",
        color=discord.Color.blue()
    )
    embed.add_field(name="How to use me", value="• Mention me with @bot then your question\n• Reply to any of my messages", inline=False)
    embed.add_field(name="Features", value="• Remembers conversation context\n• Responds to mentions and replies\n• Per-user conversation history", inline=False)
    await ctx.send(embed=embed)

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
        # Run the bot
        bot.run(DISCORD_TOKEN)
