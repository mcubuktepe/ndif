import gc
import inspect
import logging
import os
from typing import Dict

import torch
from pydantic import BaseModel
from pymongo import MongoClient
from ray import serve
from ray.serve import Application
from transformers import PreTrainedModel

from nnsight import LanguageModel
from nnsight.pydantics.Request import RequestModel

from ...schema.Response import ResponseModel, ResultModel
from ..util import set_cuda_env_var
from nnsight import util


@serve.deployment()
class ModelDeployment:
    def __init__(self, model_key: str, api_url: str, database_url: str):

        set_cuda_env_var()

        self.model_key = model_key
        self.api_url = api_url
        self.database_url = database_url

        self.model = LanguageModel(self.model_key, device_map="auto", dispatch=True)

        self.db_connection = MongoClient(self.database_url)

        self.logger = logging.getLogger(__name__)

    def __call__(self, request: RequestModel):

        try:

            request.compile()

            output = self.model.interleave(
                self.model._execute,
                request.intervention_graph,
                *request.batched_input,
                **request.kwargs,
            )

            ResponseModel(
                id=request.id,
                session_id=request.session_id,
                received=request.received,
                status=ResponseModel.JobStatus.COMPLETED,
                description="Your job has been completed.",
                result=ResultModel(
                    id=request.id,
                    # Move all copied data to cpu
                    saves={
                        name: util.apply(
                            node.value, lambda x: x.detach().cpu(), torch.Tensor
                        )
                        for name, node in request.intervention_graph.nodes.items()
                        if node.value is not inspect._empty
                    },
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
        del output

        self.model._model.zero_grad()

        gc.collect()

        torch.cuda.empty_cache()

    async def status(self):

        model: PreTrainedModel = self.model._model

        return {
            "config_json_string": model.config.to_json_string(),
            "repo_id": model.config._name_or_path,
        }

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
