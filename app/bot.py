import os
import discord 
import requests
from dotenv import load_dotenv
load_dotenv()
TOKEN = os.getenv('DISCORD_TOKEN')

class ProfanityBotClient(discord.Client):

    def check_profanity(self, text):
        url = "https://www.purgomalum.com/service/containsprofanity?text={}".format(text)

        payload = {}
        headers= {}

        response = requests.request("GET", url, headers=headers, data = payload)

        return (response.text)

    async def on_ready(self):
        print("Connected")

    async def on_message(self,message):
        if message.author.id == self.user.id:
            return 

        channel = message.channel 
        profanity = self.check_profanity(message.content)
        if profanity == 'true':
            # Flag profanity and store into ProfDB
            await channel.send("Profanity detected! Incident reported.")
        else:
            pass


ProfanityBotClient().run(TOKEN)
