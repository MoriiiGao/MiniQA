import sys
import os
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from typing import Optional, Union, List, Dict
import asyncio
from transformers import AutoTokenizer, AutoModelForCausalLM
# from llm.base import LLMInterface

