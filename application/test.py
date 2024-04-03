import sys
import os
sys.path.insert(1, os.path.expanduser('~') + '/LitArt/')

import streamlit as st
import time
import requests
from streamlit_lottie import st_lottie, st_lottie_spinner
import json
# import PyPDF2
import io
import torch

from T5.scripts.t5_inference import summarize_t5
from Pegasus.scripts.pegasus_inference import summarize_pegasus
from BART.scripts.bart_inference import summarize_bart


print(summarize_bart('''Sherlock Holmes, created by Sir Arthur Conan Doyle, is a fictional detective who first appeared in print in 1887. Holmes is renowned for his astute logical reasoning, keen observation skills, and his ability to solve complex mysteries. His adventures, chronicled by his loyal friend and narrator, Dr. John H. Watson, have captivated readers for over a century.

Holmes is depicted as residing at 221B Baker Street, London, where he conducts his investigations. He is described as tall, lean, and possessing a hawk-like nose. He is often seen wearing a deerstalker hat and smoking a pipe. Holmes is portrayed as a highly intelligent and eccentric individual, with a keen interest in chemistry, forensic science, and martial arts.

The character of Sherlock Holmes is known for his brilliant deductive reasoning. He often solves cases by observing minute details that others overlook and drawing logical conclusions from them. Holmes famously stated, "Once you eliminate the impossible, whatever remains, no matter how improbable, must be the truth."

Holmes's most famous cases include "A Study in Scarlet," "The Sign of Four," "The Adventures of Sherlock Holmes," "The Memoirs of Sherlock Holmes," "The Hound of the Baskervilles," and "The Return of Sherlock Holmes." These stories feature a wide range of mysteries, from murders and thefts to espionage and blackmail.'''))