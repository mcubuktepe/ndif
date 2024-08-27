import gc
import logging
import os
from functools import partial
from typing import Any, Dict

import torch
from pydantic import BaseModel
from pymongo import MongoClient
from ray import serve
from ray.serve import Application
from transformers import PreTrainedModel, AutoConfig

from nnsight.models.mixins import RemoteableMixin
from nnsight.schema.Request import RequestModel
from nnsight import LanguageModel

from ...schema.Response import ResponseModel, ResultModel
from ..util import set_cuda_env_var, update_nnsight_print_function
ray_serve_logger = logging.getLogger("ray.serve")

class MockSAE:
    def __init__(self):
        pass

    def encode(self, x):
        return 2*x
    def decode(self, x):
        return 2*x

@serve.deployment()
class ModelDeployment:
    def __init__(self, model_key: str, api_url: str, database_url: str):

        set_cuda_env_var()

        # Set attrs
        self.model_key = model_key
        self.api_url = api_url
        self.database_url = database_url

        # Load and dispatch model based on model key.
        # The class returned could be any model type.
        model_str = "meta-llama/Meta-Llama-3-8B-Instruct"
        config = AutoConfig.from_pretrained(model_str)
        config.attn_implementation = ("flash_attention_2",)

        self.model = LanguageModel(model_str, device_map='cuda', torch_dtype=torch.bfloat16,config=config)

        self.model.sae = MockSAE()
        # self.model = RemoteableMixin.from_model_key(
        #     self.model_key, device_map="auto", dispatch=True
        # )
        # Make model weights non trainable / no grad.
        self.model._model.requires_grad_(False)
        self.model.dispatch_model()
        # Clear cuda cache after model load.
        torch.cuda.empty_cache()

        # Init DB connection.
        self.db_connection = MongoClient(self.database_url)

        # Init logger
        self.logger = logging.getLogger(__name__)

        self.running = False

    def __call__(self, request: RequestModel):
        ray_serve_logger.error("ray deployments.model.modeldeployment line 69")
        # Send RUNNING response.
        ResponseModel(
            id=request.id,
            session_id=request.session_id,
            received=request.received,
            status=ResponseModel.JobStatus.RUNNING,
            description="Your job has started running.",
        ).log(self.logger).save(self.db_connection).blocking_response(
            self.api_url
        )
        ray_serve_logger.error("ray deployments.model.modeldeployment line 80")
        local_result = None

        try:
            ray_serve_logger.error("ray deployments.model.modeldeployment line 84")
            # Changes the nnsight intervention graph function to respond via the ResponseModel instead of printing.
            update_nnsight_print_function(
                partial(
                    self.log_to_user,
                    params={
                        "id": request.id,
                        "session_id": request.session_id,
                        "received": request.received,
                    },
                )
            )

            self.running = True
            ray_serve_logger.error(request)
            ray_serve_logger.error("ray deployments.model.modeldeployment line 99")

            # Deserialize request
            obj = request.deserialize(self.model)
            ray_serve_logger.error(obj)
            ray_serve_logger.error(dir(obj))
            ray_serve_logger.error(type(obj))

            ray_serve_logger.error("ray deployments.model.modeldeployment line 106")

            # Execute object.

            local_result = obj.local_backend_execute()
            ray_serve_logger.error("ray deployments.model.modeldeployment line 116")

            # print(b)
            ray_serve_logger.error("value")            

            #ray_serve_logger.error(obj.remote_backend_postprocess_result(local_result))            
            # Send COMPELTED response.
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
            ).log(self.logger).save(self.db_connection).blocking_response(
                self.api_url
            )
            ray_serve_logger.error("ray deployments.model.modeldeployment line 137")
            ray_serve_logger.error(self.api_url)

        except Exception as exception:

            ResponseModel(
                id=request.id,
                session_id=request.session_id,
                received=request.received,
                status=ResponseModel.JobStatus.ERROR,
                description=str(exception),
            ).log(self.logger).save(self.db_connection).blocking_response(
                self.api_url
            )

        finally:

            self.running = False

        del request
        del local_result

        self.model._model.zero_grad()

        # gc.collect()

        # torch.cuda.empty_cache()

    def log_to_user(self, data: Any, params: Dict[str, Any]):

        ResponseModel(
            **params,
            status=ResponseModel.JobStatus.LOG,
            description=str(data),
        ).log(self.logger).save(self.db_connection).blocking_response(
            self.api_url
        )

    async def status(self):

        model: PreTrainedModel = self.model._model

        return {
            "config_json_string": model.config.to_json_string(),
            "repo_id": model.config._name_or_path,
        }

    # # Ray checks this method and restarts replica if it raises an exception
    # def check_health(self):

    #     if not self.running:
    #         torch.cuda.empty_cache()

    def model_size(self) -> float:

        mem_params = sum(
            [
                param.nelement() * param.element_size()
                for param in self.model._model.parameters()
            ]
        )
        mem_bufs = sum(
            [
                buf.nelement() * buf.element_size()
                for buf in self.model._model.buffers()
            ]
        )
        mem_gbs = (mem_params + mem_bufs) * 1e-9

        return mem_gbs


class ModelDeploymentArgs(BaseModel):

    model_key: str
    api_url: str
    database_url: str


def app(args: ModelDeploymentArgs) -> Application:

    return ModelDeployment.bind(**args.model_dump())
