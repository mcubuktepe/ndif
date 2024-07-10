import logging

from pydantic import BaseModel
from pymongo import MongoClient
from ray import serve
from ray.serve import Application
from ray.serve.handle import DeploymentHandle
try:
    from slugify import slugify
except:
    pass

from nnsight.pydantics.Request import RequestModel

from ...schema.Response import ResponseModel


@serve.deployment()
class RequestDeployment:
    def __init__(self, ray_dashboard_url: str, api_url: str, database_url: str):

        self.ray_dashboard_url = ray_dashboard_url
        self.api_url = api_url
        self.database_url = database_url

        self.db_connection = MongoClient(self.database_url)

        self.logger = logging.getLogger(__name__)

    async def __call__(self, request: RequestModel):

        try:

            model_key = f"Model:{slugify(request.repo_id)}"

            app_handle = self.get_ray_app_handle(model_key)

            app_handle.remote(request)

            ResponseModel(
                id=request.id,
                session_id=request.session_id,
                received=request.received,
                status=ResponseModel.JobStatus.APPROVED,
                description="Your job was approved and is waiting to be run.",
            ).log(self.logger).save(self.db_connection).blocking_response(self.api_url)

        except Exception as exception:
            ResponseModel(
                id=request.id,
                session_id=request.session_id,
                received=request.received,
                status=ResponseModel.JobStatus.ERROR,
                description=str(exception),
            ).log(self.logger).save(self.db_connection).blocking_response(self.api_url)

    def get_ray_app_handle(self, name: str) -> DeploymentHandle:

        return serve.get_app_handle(name)


class RequestDeploymentArgs(BaseModel):

    ray_dashboard_url: str
    api_url: str
    database_url: str


def app(args: RequestDeploymentArgs) -> Application:
    return RequestDeployment.bind(**args.model_dump())
