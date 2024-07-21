
import os
from fastapi import Request, FastAPI
from fastapi.logger import logger
from huggingface_hub import hf_hub_download
from llama_cpp import Llama


app = FastAPI()

@app.on_event("startup")
def startup():
    logger.info('Application start up')
    try :
        file_path = hf_hub_download(repo_id="TheBloke/Llama-2-13B-GGUF", filename="llama-2-13b.Q2_K.gguf")

        app.llm = Llama(
            model_path=file_path,
            # n_gpu_layers=-1, # Uncomment to use GPU acceleration
            # seed=1337, # Uncomment to set a specific seed
            # n_ctx=2048, # Uncomment to increase the context window
        )
        print("Model loaded")

    except Exception as e:
        print(e)

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.get("/health", status_code=200)
def health():
    return {}


@app.post("/predict")
async def get_body(request: Request):

    request = await request.json()
    if type(request) is list:
        request = request[0]
    if type(request) is dict:
        request = request['data']

    print (request)
    model_input = request


    output = app.llm(
            model_input, # Prompt
            max_tokens=20, # Generate up to 32 tokens, set to None to generate up to the end of the context window
            stop=["Q:", "\n"], # Stop generating just before the model would generate a new question
            echo=True # Echo the prompt back in the output
        ) # Generate a completion, can also call create_completion
    
    return {"predictions": output }












