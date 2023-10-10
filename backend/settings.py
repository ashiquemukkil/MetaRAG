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
