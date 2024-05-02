import openai
import sys
import os


os.environ["OPENAI_API_KEY"] = "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"

sys.path.append("../..")

from dotenv import load_dotenv, find_dotenv

_ = load_dotenv(find_dotenv())  # read local .env file

openai.api_key = os.environ["OPENAI_API_KEY"]

try:
    os.mkdir("./docs")
except:
    pass
