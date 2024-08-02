import gc
import logging
import os
from typing import Dict

import torch
from prometheus_client import Counter, Gauge, start_http_server
from pydantic import BaseModel
from pymongo import MongoClient
from ray import serve
from ray.serve import Application
from torch.cuda import max_memory_allocated, reset_peak_memory_stats
from transformers import PreTrainedModel

from nnsight.models.mixins import RemoteableMixin
from nnsight.schema.Request import RequestModel

from ...schema.Response import ResponseModel, ResultModel
from ..util import set_cuda_env_var


@serve.deployment()
class ModelDeployment:
    def __init__(self, model_key: str, api_url: str, database_url: str):

        set_cuda_env_var()

        self.model_key = model_key
        self.api_url = api_url
        self.database_url = database_url

        self.model = RemoteableMixin.from_model_key(
            self.model_key, device_map="auto", dispatch=True
        )

        self.db_connection = MongoClient(self.database_url)

        self.logger = logging.getLogger(__name__)
        self.prometheus = start_http_server(4321)
        self.gauge = Gauge('gpu_peak_memory', 'Approximation of peak GPU VRAM during a job')

    def __call__(self, request: RequestModel):

        try:

            # Deserialize request
            obj = request.deserialize(self.model)

            # TODO: Figure out which device was used (currently can just assume device 0)
            device = 'cuda:0'
            reset_peak_memory_stats(device)
            
            # Execute object.
            local_result = obj.local_backend_execute()
            max_memory = max_memory_allocated(device)

            # TODO: Convert memory to better format
            self.gauge.set(max_memory)

            ResponseModel(
                id=request.id,
                session_id=request.session_id,
                received=request.received,
                status=ResponseModel.JobStatus.COMPLETED,
                description="Your job has been completed.",
                result=ResultModel(
                    id=request.id,
                    value=obj.remote_backend_postprocess_result(local_result),
                ),
            ).log(self.logger).save(self.db_connection).blocking_response(self.api_url)

        except Exception as exception:

            ResponseModel(
                id=request.id,
                session_id=request.session_id,
                received=request.received,
                status=ResponseModel.JobStatus.ERROR,
                description=str(exception),
            ).log(self.logger).save(self.db_connection).blocking_response(self.api_url)

        del request
        del local_result

        self.model._model.zero_grad()

        gc.collect()

        torch.cuda.empty_cache()

    async def status(self):

        model: PreTrainedModel = self.model._model

        return model.config.to_json_string()

    # Ray checks this method and restarts replica if it raises an exception
    def check_health(self):

        for device in range(torch.cuda.device_count()):
            torch.cuda.mem_get_info(device)

    def model_size(self) -> float:

        mem_params = sum(
            [
                param.nelement() * param.element_size()
                for param in self.model._model.parameters()
            ]
        )
        mem_bufs = sum(
            [buf.nelement() * buf.element_size() for buf in self.model._model.buffers()]
        )
        mem_gbs = (mem_params + mem_bufs) * 1e-9

        return mem_gbs


class ModelDeploymentArgs(BaseModel):

    model_key: str
    api_url: str
    database_url: str


def app(args: ModelDeploymentArgs) -> Application:

    return ModelDeployment.bind(**args.model_dump())
