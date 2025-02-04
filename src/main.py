from fastapi import FastAPI
from enum import Enum
from pynvml_utils import nvidia_smi
from fastapi.middleware.cors import CORSMiddleware
import psutil


app = FastAPI()


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ModelName(str, Enum):
    alexnet = "alexnet"
    resnet = "resnet"
    lenet = "lenet"


fake_items_db = [{"item_name": "Foo"}, {"item_name": "Bar"}, {"item_name": "Baz"}]


@app.get("/items/")
async def read_item(skip: int = 0, limit: int = 10):
    return fake_items_db[skip : skip + limit]


@app.get("/users/me")
async def read_user_me():
    return {"user_id": "the current user"}


@app.get("/models/{model_name}")
async def get_model(model_name: ModelName):
    if model_name is ModelName.alexnet:
        return {"model_name": model_name, "message": "Deep Learning FTW!"}

    if model_name.value == "lenet":
        return {"model_name": model_name, "message": "LeCNN all the images"}

    return {"model_name": model_name, "message": "Have some residuals"}


def nvidia_info():
    nvsmi = nvidia_smi.getInstance()

    output = nvsmi.DeviceQuery(
        'memory.total, memory.used, memory.free, utilization.gpu, utilization.memory'
    )

    memory = output['gpu'][0]['fb_memory_usage']
    utilization = output['gpu'][0]['utilization']
    outstring = {
        'vram_total': f"{memory['total']} {memory['unit']}",
        'vram_used': f"{memory['used']} {memory['unit']}",
        'vram_free': f"{memory['free']} {memory['unit']}",
        'gpu_util': f"{utilization['gpu_util']:3d}{utilization['unit']}",
        'mem_util': f"{utilization['memory_util']:3d}{utilization['unit']}",
    }
    return outstring


@app.get("/gpu")
def get_gpu_usage_all():
    return nvidia_info()


@app.get("/gpu/{item}")
def get_gpu_usage(item):
    outstring = nvidia_info()

    return outstring[item]


if __name__ == '__main__':
    app = FastAPI()
