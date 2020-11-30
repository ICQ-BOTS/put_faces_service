import asyncio
from aiohttp import web
import configparser
import os
import os.path
import cv2
import base64
import draw
import shutil
from mailru_im_async_bot.bot import Bot
from concurrent.futures import ProcessPoolExecutor
import aiofiles
import io
import nest_asyncio
import traceback
import utils
import json

nest_asyncio.apply()

SCRIPT_PATH = os.path.dirname(os.path.realpath(__file__))
ENCODING = "utf-8"

SHAWARMA_IMAGES_DIRECTORY = os.path.join(SCRIPT_PATH, "shawarma")
COMMON_SHAWARMA_IMAGES_DIRECTORY = os.path.join(SHAWARMA_IMAGES_DIRECTORY, "common")
ST_P_SHAWARMA_IMAGES_DIRECTORY = os.path.join(SHAWARMA_IMAGES_DIRECTORY, "st_p")
WATERMAKS_DIRECTORY = os.path.join(SCRIPT_PATH, "watermarks")
TEMP_DIRECTORY = os.path.join(SCRIPT_PATH, "temp")


config = configparser.ConfigParser()
config.read(os.path.join(SCRIPT_PATH, "config.ini"))

HOST_IP = config.get("main", "host")
HOST_PORT = config.get("main", "port")
PROCESSES_COUNT = int(config.get("main", "processes_count"))
TRASH_CHAT = config.get("main", "trash_chat")

# BOTS MAIN CONFIG
VERSION = "0.0.1"
HASH_ = None
POLL_TIMEOUT_S = int(config.get("icq_bot", "poll_time_s"))
REQUEST_TIMEOUT_S = int(config.get("icq_bot", "request_timeout_s"))
TASK_TIMEOUT_S = int(config.get("icq_bot", "task_timeout_s"))
TASK_MAX_LEN = int(config.get("icq_bot", "task_max_len"))


def get_next_file_name(template):
    i = 0
    while os.path.exists(template % (i)):
        i += 1
    return template % (i)


def get_reference_to_file(file_id):
    return "https://files.icq.net/get/%s" % file_id


bots_cache = dict()


async def send_callback_image_async(
    bot_access_token,
    bot_name,
    callback_chat_id,
    callback_message_id,
    callback_text,
    callback_buttons,
    image_bytes=None,
):
    bot = Bot(
        token=bot_access_token,
        version=VERSION,
        name=bot_name,
        poll_time_s=POLL_TIMEOUT_S,
        request_timeout_s=REQUEST_TIMEOUT_S,
        task_max_len=TASK_MAX_LEN,
        task_timeout_s=TASK_TIMEOUT_S,
    )
    if not image_bytes is None:
        image_bytes_stream = io.BytesIO(image_bytes)
        image_bytes_stream.seek(0)
        image_bytes_stream.name = "generated_image.jpg"

        upload_response = await bot.send_file(
            chat_id=TRASH_CHAT, file=image_bytes_stream
        )
        message_image = upload_response.get("fileId", None)
    else:
        message_image = None

    if not message_image is None:
        if callback_text:
            callback_text = callback_text + " " + get_reference_to_file(message_image)
        else:
            callback_text = get_reference_to_file(message_image)
    await bot.edit_text(
        msg_id=callback_message_id,
        chat_id=callback_chat_id,
        text=callback_text,
        inline_keyboard_markup=callback_buttons,
    )
    await bot.stop()


def send_callback_image(
    bot_access_token,
    bot_name,
    callback_chat_id,
    callback_message_id,
    callback_text,
    callback_buttons,
    image_bytes=None,
):
    import bot.bot

    bot = bot.bot.Bot(token=bot_access_token)
    if not image_bytes is None:
        image_bytes_stream = io.BytesIO(image_bytes)
        image_bytes_stream.seek(0)
        image_bytes_stream.name = "generated_image.jpg"

        upload_response = bot.send_file(
            chat_id=TRASH_CHAT, file=image_bytes_stream
        ).json()
        message_image = upload_response.get("fileId", None)
    else:
        message_image = None

    if not message_image is None:
        if callback_text:
            callback_text = callback_text + " " + get_reference_to_file(message_image)
        else:
            callback_text = get_reference_to_file(message_image)
    bot.edit_text(
        msg_id=callback_message_id,
        chat_id=callback_chat_id,
        text=callback_text,
        inline_keyboard_markup=callback_buttons,
    )
    bot.stop()


def empty():
    pass


def put_face_to_shawarma(
    bot_access_token,
    bot_name,
    callback_chat_id,
    callback_message_id,
    callback_text,
    callback_buttons,
    face_image_file_path,
    shawarma_image_file_path,
    is_shawarma_temp,
    watermark,
):
    try:
        loop = asyncio.get_event_loop()

        # Read cached images
        face_image_file_path = face_image_file_path.replace("\\", "/")
        face_image = cv2.imread(face_image_file_path)

        shawarma_image = cv2.imread(shawarma_image_file_path)
        # Remove temporary files
        os.remove(face_image_file_path)
        if is_shawarma_temp:
            os.remove(shawarma_image_file_path)

        put_to_shawarma_result = draw.put_to_shawarma(face_image, shawarma_image)

        error_code = put_to_shawarma_result[0]
        if error_code != 0:
            error_message = f"Что-то пошло не так :("
            error_image = shawarma_image

            if error_code == 1:
                error_message = (
                    "Я не нашел на фото лица, отправь, пожалуйста, другое фото"
                )
            elif error_code == 2:
                error_message = (
                    "Я не обнаружил шаурму, отправь, пожалуйста, другое фото"
                )
                error_image = None

            if not error_image is None:
                if not watermark is None:
                    draw.add_watermark(error_image, *watermark)
                is_success, buffer = cv2.imencode(".jpg", error_image)
            else:
                buffer = None
            send_callback_image(
                bot_access_token,
                bot_name,
                callback_chat_id,
                callback_message_id,
                callback_text=error_message,
                callback_buttons=callback_buttons,
                image_bytes=buffer,
            )
            return

        result_image = put_to_shawarma_result[1]

        if not watermark is None:
            draw.add_watermark(result_image, *watermark)

        is_success, buffer = cv2.imencode(".jpg", result_image)

        send_callback_image(
            bot_access_token,
            bot_name,
            callback_chat_id,
            callback_message_id,
            callback_text,
            callback_buttons,
            image_bytes=buffer,
        )
    except:
        traceback.print_exc()


async def close_server():
    print("Received stop command")
    await asyncio.sleep(0.5)
    raise web.GracefulExit()


async def server_stop(request):
    loop.create_task(close_server())
    return web.Response(text="OK")


def select_with_condition(array, condition, arg_index=0):
    arg_value = condition[arg_index]

    if arg_index == len(condition) - 1:
        generator = array
    else:
        generator = select_with_condition(array, condition, arg_index + 1)

    if arg_value is None:
        for element in generator:
            yield element
    else:
        for element in generator:
            if element[arg_index] == arg_value:
                yield element


def parse_arg(value, arg_array):
    if not value:
        return None
    else:
        return arg_array.index(value)


async def shawarma_put(request):
    data = await request.read()
    args = json.loads(data)

    bot_info = args.get("bot_info", None)
    bot_access_token = bot_info[0]
    bot_name = bot_info[1]

    callback_message_id = args.get("callback_message_id", None)
    callback_chat_id = args.get("callback_chat_id", None)
    watermark_name = args.get("watermark", None)

    watermark = None
    if not watermark_name is None:
        watermark = watermarks.get(watermark_name, None)

    face_image_bytes_string = args.get("face_image", None)
    shawarma_image_bytes_string = args.get("shawarma_image", None)
    callback_addition = args.get("callback_addition", None)

    callback_text = None
    callback_buttons = None
    if not callback_addition is None:
        callback_text = callback_addition.get("text", None)
        callback_buttons = callback_addition.get("buttons", None)
    shawarma_image_bytes = None

    # Default shawarma
    if shawarma_image_bytes_string is None:
        shawarma_generator = args.get("shawarma_generator", None)
        is_shawarma_temp = False
        shawarma_number = shawarma_generator[0]
        shawarma_class = parse_arg(shawarma_generator[1], shawarma_classes)
        shawarma_color = parse_arg(shawarma_generator[2], shawarma_colors)

        selected_elements = list(
            select_with_condition(shawarma_list, (None, shawarma_class, shawarma_color))
        )
        shawarma_file_path = selected_elements[
            shawarma_number % len(selected_elements)
        ][0]
    # Custom shawarma
    else:
        is_shawarma_temp = True
        shawarma_file_path = get_next_file_name(os.path.join(TEMP_DIRECTORY, "tmp_%d"))
        async with aiofiles.open(shawarma_file_path, "wb") as f:
            shawarma_image_bytes = base64.b64decode(shawarma_image_bytes_string)
            await f.write(shawarma_image_bytes)
            await f.close()

    if face_image_bytes_string:  # With face image
        face_temp_file_path = get_next_file_name(os.path.join(TEMP_DIRECTORY, "tmp_%d"))
        async with aiofiles.open(face_temp_file_path, "wb") as f:
            await f.write(base64.b64decode(face_image_bytes_string))
            await f.close()

        # Both images are loaded

        processes_pool.submit(
            put_face_to_shawarma,
            bot_access_token,
            bot_name,
            callback_chat_id,
            callback_message_id,
            callback_text,
            callback_buttons,
            face_temp_file_path,
            shawarma_file_path,
            is_shawarma_temp,
            watermark,
        )
    else:  # Without face image
        if shawarma_image_bytes is None:
            async with aiofiles.open(shawarma_file_path, "rb") as f:
                shawarma_image_bytes = await f.read()

        if is_shawarma_temp:
            os.remove(shawarma_image_file_path)

        if not shawarma_image_bytes is None:
            if not watermark is None:
                shawarma_image = utils.image_from_bytes(shawarma_image_bytes)
                draw.add_watermark(shawarma_image, *watermark)
                is_success, shawarma_image_bytes = cv2.imencode(".jpg", shawarma_image)

        # With custom shawarma image
        await send_callback_image_async(
            bot_access_token,
            bot_name,
            callback_chat_id,
            callback_message_id,
            callback_text,
            callback_buttons,
            image_bytes=shawarma_image_bytes,
        )

    result = {"status": "OK"}

    return web.json_response(result)


shawarma_list = []
shawarma_classes = ["common", "st_p"]
shawarma_colors = ["red", "green", "orange", "black", "white"]


def get_shawarma_color_by_file_name(shawarma_file_name):
    for index in range(len(shawarma_colors)):
        if shawarma_colors[index] in shawarma_file_name:
            return index
    return None


if __name__ == "__main__":
    # Load shawarma

    for obj in os.listdir(COMMON_SHAWARMA_IMAGES_DIRECTORY):
        full_path = os.path.join(COMMON_SHAWARMA_IMAGES_DIRECTORY, obj)
        shawarma_list.append((full_path, 0, get_shawarma_color_by_file_name(obj)))

    for obj in os.listdir(ST_P_SHAWARMA_IMAGES_DIRECTORY):
        full_path = os.path.join(ST_P_SHAWARMA_IMAGES_DIRECTORY, obj)
        shawarma_list.append((full_path, 1, get_shawarma_color_by_file_name(obj)))

    # prepare temp directory
    if os.path.exists(TEMP_DIRECTORY):
        shutil.rmtree(TEMP_DIRECTORY, ignore_errors=True)

    os.makedirs(TEMP_DIRECTORY)

    # Watermarks preload
    watermarks = {}
    for obj in os.listdir(WATERMAKS_DIRECTORY):
        full_path = os.path.join(WATERMAKS_DIRECTORY, obj)
        watermarks[os.path.splitext(obj)[0]] = utils.bgra_to_bgr_and_mask(
            cv2.imread(full_path, cv2.IMREAD_UNCHANGED)
        )
    processes_pool = ProcessPoolExecutor(PROCESSES_COUNT)

    loop = asyncio.get_event_loop()
    app = web.Application(loop=loop, client_max_size=100000000)
    app.router.add_post("/shawarma_put", shawarma_put)
    app.router.add_post("/stop", server_stop)

    processes_pool.map(empty)

    try:
        web.run_app(app, host=HOST_IP, port=int(HOST_PORT))
    except web.GracefulExit:
        print("server was stopped")
