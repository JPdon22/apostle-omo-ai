from telethon.sync import TelegramClient

api_id = 31439354
api_hash = "783902e19292dfc5b7d9b3121b8406ba"

channel_username = "apostleomo_messages"

client = TelegramClient("apostleomo_session", api_id, api_hash)

async def main():
    print("Reading channel messages...\n")

    async for message in client.iter_messages(channel_username, limit=10):
        print("MESSAGE:")
        print(message.text)
        print("-" * 50)

with client:
    client.loop.run_until_complete(main())