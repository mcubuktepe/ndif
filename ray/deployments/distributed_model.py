import gc
import logging
import socket
from datetime import timedelta
from typing import Dict

import ray
import torch
import torch.distributed
from pymongo import MongoClient
from ray import serve
from ray.serve import Application
from transformers import PreTrainedModel

from nnsight.models.mixins import RemoteableMixin
from nnsight.schema.Request import RequestModel

from ...schema.Response import ResponseModel, ResultModel
from ..distributed.parallel_dims import ParallelDims
from ..distributed.tensor_parallelism import parallelize_model
from ..distributed.util import load_hf_model_from_cache
from .model import ModelDeploymentArgs


@serve.deployment(ray_actor_options={"num_gpus": 1})
class ModelDeployment:
    def __init__(
        self,
        model_key: str,
        api_url: str,
        database_url: str,
        torch_distributed_address: str,
        torch_distributed_port: int,
        torch_distributed_world_size: int,
        torch_distributed_world_rank: int,
        torch_distributed_world_timeout_seconds: int,
        data_parallelism_size: int,
        tensor_parallelism_size: int,
        pipeline_parallelism_size: int,
    ):

        self.replica_context = serve.get_replica_context()

        self.model_key = model_key
        self.api_url = api_url
        self.database_url = database_url
        self.torch_distributed_address = torch_distributed_address
        self.torch_distributed_port = torch_distributed_port
        self.torch_distributed_world_size = torch_distributed_world_size
        self.torch_distributed_world_rank = torch_distributed_world_rank
        self.torch_distributed_world_timeout_seconds = (
            torch_distributed_world_timeout_seconds
        )
        self.data_parallelism_size = data_parallelism_size
        self.tensor_parallelism_size = tensor_parallelism_size
        self.pipeline_parallelism_size = pipeline_parallelism_size

        self.model = RemoteableMixin.from_model_key(self.model_key)

        self.device = torch.device(f"cuda:{torch.cuda.current_device()}") 

        print("device", str(self.device))

        self.logger = logging.getLogger(__name__)

        self.head = torch_distributed_world_rank == 0

        print("head", self.head)
        print("taddress", self.torch_distributed_address)

        if self.head:

            self.db_connection = MongoClient(self.database_url)

            if self.torch_distributed_address is None:

                ip_address = ray.get_runtime_context().worker.node_ip_address

                self.torch_distributed_address = (
                    f"tcp://{ip_address}:{self.torch_distributed_port}"
                )

            self.worker_deployments = []

            for worker_world_rank in range(1, self.torch_distributed_world_size):

                distributed_model_deployment_args = DistributedModelDeploymentArgs(
                    model_key=self.model_key,
                    api_url=self.api_url,
                    database_url=self.database_url,
                    torch_distributed_address=self.torch_distributed_address,
                    torch_distributed_world_size=self.torch_distributed_world_size,
                    torch_distributed_world_rank=worker_world_rank,
                    torch_distributed_world_timeout_seconds=self.torch_distributed_world_timeout_seconds,
                    tensor_parallelism_size=self.tensor_parallelism_size,
                    data_parallelism_size=self.data_parallelism_size,
                    pipeline_parallelism_size=self.pipeline_parallelism_size,
                )

                worker_application = ModelDeployment.bind(
                    **distributed_model_deployment_args.model_dump()
                )

                worker_deployment = serve.run(
                    worker_application,
                    name=f"{self.replica_context.app_name}-{worker_world_rank}",
                    route_prefix=f"/{self.replica_context.app_name}-{worker_world_rank}",
                )

                self.worker_deployments.append(worker_deployment)

            for worker_deployment in self.worker_deployments:

                worker_deployment.init_distributed.remote()

            self.init_distributed()

    def init_distributed(self):

        torch.distributed.init_process_group(
            "nccl",
            init_method=self.torch_distributed_address,
            timeout=timedelta(self.torch_distributed_world_timeout_seconds),
            world_size=self.torch_distributed_world_size,
            rank=self.torch_distributed_world_rank,
            device_id=self.device,
        )

        print("dist inited")

        parallel_dims = ParallelDims(
            dp=self.data_parallelism_size,
            tp=self.tensor_parallelism_size,
            pp=self.pipeline_parallelism_size,
            world_size=self.torch_distributed_world_size,
            enable_loss_parallel=False,
        )

        world_mesh = parallel_dims.build_mesh(device_type=f"cuda")

        parallelize_model(self.model._model, self.model._model.config._name_or_path, world_mesh["tp"])
        print("parallelized")

        load_hf_model_from_cache(
            self.model._model, self.model._model.config._name_or_path
        )

        print("loaded")

        self.model._dispatched = True

    def __call__(self, request: RequestModel):

        try:

            if self.head:

                for worker_deployment in self.worker_deployments:

                    worker_deployment.remote(request)

            # Deserialize request
            obj = request.deserialize(self.model)

            # Execute object.
            local_result = obj.local_backend_execute()

            if self.head:

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

        except Exception as exception:

            if self.head:

                ResponseModel(
                    id=request.id,
                    session_id=request.session_id,
                    received=request.received,
                    status=ResponseModel.JobStatus.ERROR,
                    description=str(exception),
                ).log(self.logger).save(self.db_connection).blocking_response(
                    self.api_url
                )

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
        torch.cuda.synchronize()


class DistributedModelDeploymentArgs(ModelDeploymentArgs):

    torch_distributed_address: str = None
    torch_distributed_port: int = None
    torch_distributed_world_rank: int = 0

    torch_distributed_world_size: int
    torch_distributed_world_timeout_seconds: int

    data_parallelism_size: int = 1
    tensor_parallelism_size: int = 1
    pipeline_parallelism_size: int = 1


def app(args: DistributedModelDeploymentArgs) -> Application:
    return ModelDeployment.bind(**args.model_dump())
