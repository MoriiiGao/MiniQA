import os
import torch
from fastapi import FastAPI, Request
from pydantic import BaseModel
from typing import List, Optional
from vllm import LLM, SamplingParams

