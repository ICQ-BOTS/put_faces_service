import asyncio
import aiohttp
from multiprocessing import Pool, freeze_support
from concurrent.futures import ProcessPoolExecutor
import time
import os
import os.path
import configparser


SCRIPT_PATH = os.path.dirname(os.path.realpath(__file__))

config = configparser.ConfigParser()
config.read(os.path.join(SCRIPT_PATH, "config.ini"))

HOST_IP = config.get("main", "host")
HOST_PORT = config.get("main", "port")

async def main():
    session = aiohttp.ClientSession()
    #response = await session.post('http://127.0.0.1:1518/shawarma_put', json = {'name' : 'Android-1'})
    response = await session.post(f'http://{HOST_IP}:{HOST_PORT}/stop')
    print(await response.text())
    await session.close()

if __name__ == '__main__':
    ioloop = asyncio.get_event_loop()
    ioloop.run_until_complete(main())