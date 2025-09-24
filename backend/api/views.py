import base64
import logging
import os
import time

from django.conf import settings
from django.contrib.auth.models import Group, User
from django.utils import timezone
from django_rq import get_queue
from django_rq.jobs import Job
from drf_yasg.utils import swagger_auto_schema
from rest_framework import status
from rest_framework.decorators import api_view
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response
from rest_framework.views import APIView
from rest_framework_simplejwt.tokens import AccessToken
from rq import Callback
from rq.command import send_stop_job_command
from rq.exceptions import NoSuchJobError

from api.decorators import api_key_required
from api.models import APIKey
from api.serializers import CreateTaskSerializer, TokenSerializer
from api.tasks import batch_processing_task
from utils.config import configure_logging
from utils.utils import generate_random_password, job_queue_rank, on_success

configure_logging(
    logger_name="api_logger",
    filename="api_logs.log",
    level=logging.INFO,
)

logger = logging.getLogger("api_logger")


class HelloWorldView(APIView):
    def get(self, request):
        return Response({"message": "Hello World!"})


@swagger_auto_schema(method="post", auto_schema=None)
@api_view(["POST"])
def create_api_key(request):
    group_name = request.data.get("group_name")
    if not group_name:
        return Response(
            {"error": "Group name required"},
            status=status.HTTP_400_BAD_REQUEST,
        )
    logger.info(f"Creating API key for group: {group_name}")
    try:
        group = Group.objects.get(name=group_name.lower())
    except Group.DoesNotExist:
        logger.error("Group not found")
        return Response({"error": "Group not found"}, status=status.HTTP_404_NOT_FOUND)

    existing_api_key = APIKey.objects.filter(group=group).first()

    if existing_api_key:
        if existing_api_key.is_valid():
            logger.error("API key already exists for this group")
            return Response(
                {"error": "API key already exists for this group"},
                status=status.HTTP_400_BAD_REQUEST,
            )
        else:
            existing_api_key.key = APIKey.generate_key()
            existing_api_key.expires_at = (
                timezone.now() + settings.API_KEY_EXPIRATION_DURATION
            )
            existing_api_key.save()
            logger.info("API key updated successfully")
            return Response(
                {"api_key": existing_api_key.key}, status=status.HTTP_200_OK
            )
    api_key = APIKey.objects.create(
        group=group,
        key=APIKey.generate_key(),
        expires_at=timezone.now() + settings.API_KEY_EXPIRATION_DURATION,
    )
    logger.info("API key created successfully")
    return Response({"api_key": api_key.key}, status=status.HTTP_201_CREATED)


@swagger_auto_schema(method="post", auto_schema=None)
@api_view(["POST"])
def revoke_api_key(request):
    api_key = request.data.get("api_key")
    logger.info(f"Revoking API key: {api_key}")
    try:
        api_key_obj = APIKey.objects.get(key=api_key)
        api_key_obj.is_revoked = True
        api_key_obj.save()
        logger.info("API key revoked successfully")
        return Response(
            {"message": "API Key revoked successfully"}, status=status.HTTP_200_OK
        )
    except APIKey.DoesNotExist:
        logger.error("API key not found")
        return Response(
            {"error": "API Key not found"}, status=status.HTTP_404_NOT_FOUND
        )


class TokenView(APIView):
    @api_key_required
    def post(self, request):
        """Generate a new access token based on CUID validation

        **Parameters**:
        - `cuid`: CUID of the user (str).

        **Responses**:
        - 200: Access token generated successfully.
        - 400: Validation errors.
        - 401: Unauthorized CUID.

        **Request example**:
        ```json
        {
            "cuid": "valid_cuid",
            "email": "valid_email@orange.com"
        }
        ```

        **Response example**:
        ```json
        {
        "access_token": "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
        "expires_in": 3600
        }
        ```
        """
        logger.info("Generating access token")
        serializer = TokenSerializer(data=request.data)
        if not serializer.is_valid():
            logger.error(f"Validation failed : {serializer.errors}")
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
        cuid = serializer.validated_data.get("cuid")
        email = serializer.validated_data.get("email")
        user = User.objects.filter(username=cuid.lower()).first()
        if not user and not email:
            logger.error("Email required for new user registration")
            return Response(
                {"error": "Email required for new user registration"},
                status=status.HTTP_400_BAD_REQUEST,
            )
        if not user:
            logger.info("Creating new user")
            user = User.objects.create(username=cuid.lower(), email=email)
            user.set_password(generate_random_password())
            user.save()
            api_key_group = self.api_key.group
            api_key_group.user_set.add(user)

        token_lifetime = settings.SIMPLE_JWT.get("ACCESS_TOKEN_LIFETIME")
        access_token = AccessToken.for_user(user)
        access_token.set_exp(lifetime=token_lifetime)
        logger.info("Access token generated successfully")
        return Response(
            {
                "access_token": str(access_token),
                "expires_in": int(token_lifetime.total_seconds()),
            },
            status=status.HTTP_200_OK,
        )


class CreateTaskView(APIView):
    permission_classes = [IsAuthenticated]

    def post(self, request):
        """Create a new batch processing task

        **Parameters**:
        - `cuid`: CUID of the user (str).
        - `scope`: Scope of the batch processing (zone, list of ic01, period start, period end, filters (optional) ).
        - `prompts`: List of prompts.

        **Responses**:
        - 200: Batch creation task created successfully.
        - 400: Validation errors.
        - 401: Unauthorized CUID.
        - 500: Internal server error.

        **Request example**:
        ```json
        {
            "cuid": "valid_cuid",
            "scope": {
                "zone": "eqt"
                "ic01_list": ["ic01_1", "ic01_2", "ic01_3"],
                "period_start": "2024-01-01",
                "period_end": "2024-09-30",
                "filters": {
                    "status":"SGDG,PEND",
                    "contract_type": "CPSE",
                    "sales_country": {"value": ["country1", "country2"]},
                    "sales_region": {"value": ["region1", "region2"]},
                    "document_label": {"value": ["Legal"], filter_type: "keep"},
                    "original_file_name": {"value": ["awt"]},
                    "document_creation_date": {"start_period": "2024-06-27", "end_period": "2024-06-28"}
                }
            }
            "prompts": [
                {"name": "prompt_1", "value": "prompt_1_value"},
                {"name": "prompt_2", "value": "prompt_2_value"}
            ]
            "fields": ["typeLabel", "statusLabel", "initialCreatedBy"]
        }
        ```

        **Response example**:
        ```json
        {
            "job_id": "1234567890abcdef",
            "creation_date": "2023-10-01T12:00:00Z",
            "status": "queued",
            "queue_rank": 1
        }
        ```
        """
        logger.info("Creating a new batch processing task")
        serializer = CreateTaskSerializer(data=request.data)
        if not serializer.is_valid():
            logger.error(f"Validation failed : {serializer.errors}")
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
        cuid = serializer.validated_data.get("cuid")
        scope = serializer.validated_data.get("scope")
        prompts = serializer.validated_data.get("prompts")
        fields = serializer.validated_data.get("fields")
        if not User.objects.filter(username=cuid.lower()).exists():
            logger.error("User not found")
            return Response(
                {"error": "User not found"}, status=status.HTTP_401_UNAUTHORIZED
            )
        user = User.objects.get(username=cuid.lower())

        try:
            queue = get_queue("default")
            job = queue.enqueue(
                batch_processing_task,
                cuid=user.username,
                scope=scope,
                prompts=prompts,
                fields=fields,
                on_success=Callback(on_success),
                meta={"email_address": user.email},
                result_ttl=-1,
            )
            logger.info(
                f"Batch processing task created successfully with job id: {job.id}"
            )
            return Response(
                {
                    "job_id": job.id,
                    "creation_date": job.created_at,
                    "status": job.get_status(),
                    "queue_rank": job_queue_rank(queue, job.id),
                },
                status=status.HTTP_202_ACCEPTED,
            )
        except Exception as e:
            logger.error(
                f"Error occurred while creating batch processing task: {str(e)}"
            )
            return Response(
                {"error": f"An error occured: {str(e)}."},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )


class GetDeleteTaskView(APIView):
    permission_classes = [IsAuthenticated]

    def get(self, request, job_id: str):
        """Get status of a batch processing task by job_id

        **Parameters**:
        - `job_id`: The job id (str).

        **Responses**:
        - 200: Batch task status fetched successfully.
        - 400: Validation errors.
        - 404: Job not found.
        - 500: Internal server error.

        **Request example**:
        ```
        GET /api/batch/1234567890abcdef/
        ```

        **Response example**:
        ```json
        {
            "status": "finished",
            "queue_rank": 1
        }
        ```
        """
        logger.info(f"Fetching status for job : {job_id}")
        if not job_id:
            logger.error("Job ID is required")
            return Response(
                {"message": "Job ID is required"},
                status=status.HTTP_400_BAD_REQUEST,
            )
        queue = get_queue("default")
        try:
            job = Job.fetch(job_id, connection=queue.connection)
            logger.info("Job status fetched successfully")
            return Response(
                {
                    "status": job.get_status(),
                    "queue_rank": job_queue_rank(queue, job.id),
                },
                status=status.HTTP_200_OK,
            )
        except NoSuchJobError:
            logger.error(f"Batch {job_id} not found")
            return Response(
                {"message": f"Batch {job_id} not found"},
                status=status.HTTP_404_NOT_FOUND,
            )
        except Exception as e:
            return Response(
                {"message": f"An error occurred: {str(e)}."},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )

    def delete(self, request, job_id: str):
        """Delete a batch processing task by job_id

        **Parameters**:
        - `job_id`: The job id (str).

        **Responses**:
        - 204: Batch processing task deleted successfully.
        - 400: Validation errors.
        - 404: Job not found.
        - 500: Internal server error.

        **Request example**:
        ```
        DELETE /api/batch/1234567890abcdef/
        ```

        **Response example**:
        ```json
        {}
        ```
        """
        logger.info(f"Deleting job: {job_id}")
        if not job_id:
            logger.error("Job ID is required")
            return Response(
                {"message": "Job ID is required"},
                status=status.HTTP_400_BAD_REQUEST,
            )
        queue = get_queue("default")
        try:
            job = Job.fetch(job_id, connection=queue.connection)
            if job.is_started:
                send_stop_job_command(queue.connection, job_id)
            time.sleep(2)
            job.delete()
            logger.info(f"Batch processing task {job_id} deleted successfully")
            return Response(
                {},
                status=status.HTTP_204_NO_CONTENT,
            )

        except NoSuchJobError:
            logger.error(f"Batch {job_id} not found")
            return Response(
                {"message": f"Batch {job_id} not found"},
                status=status.HTTP_404_NOT_FOUND,
            )
        except Exception as e:
            logger.error(
                f"Error occurred while deleting batch processing task: {str(e)}"
            )
            return Response(
                {"message": f"An error occurred: {str(e)}."},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )


class TaskResultView(APIView):
    permission_classes = [IsAuthenticated]

    def get(self, request, job_id: str):
        """Get result of a batch processing task by job_id

        **Parameters**:
        - `job_id`: The job id (str).

        **Responses**:
        - 200: Batch task result fetched successfully.
        - 400: Validation errors.
        - 404: Job not found.
        - 500: Internal server error.

        **Request example**:
        ```
        GET /api/batch-result/1234567890abcdef/
        ```

        **Response example**:
        ```json
        {
            "file_name": "1234567890abcdef.xlsx",
            "file_content": "base64_encoded_content_here"
        }
        ```
        """
        logger.info(f"Fetching result for job: {job_id}")
        if not job_id:
            logger.error("Job ID is required")
            return Response(
                {"message": "Job ID is required."},
                status=status.HTTP_400_BAD_REQUEST,
            )
        queue = get_queue("default")
        data_path = os.path.join(settings.BASE_DIR, "data")
        os.makedirs(data_path, exist_ok=True)
        try:
            job = Job.fetch(job_id, connection=queue.connection)
            job_file_name = f"{job.id}.xlsx"
            job_file_path = os.path.join(data_path, job_file_name)
            if not os.path.exists(job_file_path):
                logger.error("Job file not found")
                return Response(
                    {"error": "Job file not found"}, status=status.HTTP_404_NOT_FOUND
                )
            with open(job_file_path, "rb") as excel_file:
                excel_content = excel_file.read()
            base64_content = base64.b64encode(excel_content).decode("utf-8")
            logger.info(f"Result fetched successfully for job: {job_id}")
            return Response(
                {
                    "file_name": job_file_name,
                    "file_content": base64_content,
                },
                status=status.HTTP_200_OK,
            )
        except NoSuchJobError:
            logger.error(f"Batch {job_id} not found")
            return Response(
                {"message": f"Batch {job_id} not found."},
                status=status.HTTP_404_NOT_FOUND,
            )
        except Exception as e:
            logger.error(
                f"Error occurred while fetching result for batch processing task: {str(e)}"
            )
            return Response(
                {"message": f"An error occurred: {str(e)}."},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )
