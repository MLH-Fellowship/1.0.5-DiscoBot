import os
import discord
import requests
from dotenv import load_dotenv
from pymongo import MongoClient

load_dotenv()
TOKEN = os.getenv("DISCORD_TOKEN")


class ProfanityBotClient(discord.Client):
    async def databaseOps(self, user_id, name, text):
        # establing connection
        try:
            client = MongoClient("mongodb://localhost:27017/")
            print("Connected successfully!!!")
        except:
            print("Could not connect to MongoDB")

        # Check if database exists.
        dbnames = client.list_database_names()
        if "profantor" in dbnames:
            # connecting or switching to the database
            db = client.profantor

            # creating or switching to demoCollection
            collection = db.offences
            post = {"user_id": user_id, "name": name, "text": text, "offenses": 1}
            record = collection.insert_one(post)
            print(record)

    def check_profanity(self, text):
        url = "https://www.purgomalum.com/service/containsprofanity?text={}".format(
            text
        )

        payload = {}
        headers = {}

        response = requests.request("GET", url, headers=headers, data=payload)

        return response.text

    async def on_ready(self):
        print("Connected")

    async def on_message(self, message):
        if message.author.id == self.user.id:
            return
        profanity = self.check_profanity(message.content)
        await self.databaseOps(message.author.id, message.author.name, message.content)
        channel = message.channel
        if profanity == "true":
            # Flag profanity and store into ProfDB
            await channel.send("Profanity detected! Incident reported.")
        else:
            pass


ProfanityBotClient().run(TOKEN)
