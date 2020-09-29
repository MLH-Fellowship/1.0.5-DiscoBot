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
            print("[INFO] Connected to MongoDB")
        except:
            print("[ERROR] Could not connect to MongoDB")
 
        # Check if database exists.
        dbnames = client.list_database_names()
        if "profantor" in dbnames:
            # connecting or switching to the database
            db = client.profantor

            # creating or switching to demoCollection
            collection = db.offences
            if collection.count_documents({ 'user_id': user_id }, limit = 1) != 0:
                collection.find_and_modify(query={'user_id':user_id}, update={"$inc": {'offenses': 1}}, upsert=False, full_response= True)
            else:
                post = {"user_id": user_id, "name": name, "offenses": 1}
                record = collection.insert_one(post)

    def check_profanity(self, text):
        url = "https://www.purgomalum.com/service/containsprofanity?text={}".format(text)

        payload = {}
        headers= {}

        response = requests.request("GET", url, headers=headers, data = payload)

        return (response.text)

    async def on_ready(self):
        print("[INFO] Connected to Discord")

    async def on_message(self, message):
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
