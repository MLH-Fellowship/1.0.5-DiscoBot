import os
import discord 
from dotenv import load_dotenv
load_dotenv()
TOKEN = os.getenv('DISCORD_TOKEN')

class ProfanityBotClient(discord.Client):


    async def on_ready(self):
        print("Connected")

    async def on_message(self,message):
        if message.author.id == self.user.id:
            return 

        channel = message.channel 
        #
        await channel.send("Test back")


ProfanityBotClient().run(TOKEN)
