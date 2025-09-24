import json
from datetime import datetime
from unittest.mock import MagicMock, Mock, patch

from django.contrib.auth.models import Group, User
from django.urls import reverse
from rest_framework import status
from rest_framework.test import APITestCase
from rq.exceptions import NoSuchJobError

from api.models import APIKey


class HelloWorldViewTests(APITestCase):
    def test_hello_world(self):
        url = reverse("hello world")
        response = self.client.get(url)
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(response.data, {"message": "Hello World!"})


class APIKeyViewTests(APITestCase):
    def setUp(self):
        self.group = Group.objects.create(name="test_group")
        self.create_api_key_url = reverse("create_api_key")
        self.revoke_api_key_url = reverse("revoke_api_key")

    def test_create_api_key(self):
        response = self.client.post(
            self.create_api_key_url,
            json.dumps({"group_name": "test_group"}),
            content_type="application/json",
        )
        self.assertEqual(response.status_code, status.HTTP_201_CREATED)
        self.assertIn("api_key", response.data)

    def test_revoke_api_key(self):
        api_key = APIKey.objects.create(
            group=self.group, key=APIKey.generate_key(), expires_at=None
        )
        response = self.client.post(
            self.revoke_api_key_url,
            json.dumps({"api_key": api_key.key}),
            content_type="application/json",
        )
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        api_key.refresh_from_db()
        self.assertTrue(api_key.is_revoked)

    def test_create_api_key_without_group(self):
        response = self.client.post(
            self.create_api_key_url, {}, content_type="application/json"
        )
        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)
        self.assertEqual(response.data, {"error": "Group name required"})

    def test_create_api_key_already_exists(self):
        APIKey.objects.create(
            group=self.group, key=APIKey.generate_key(), expires_at=None
        )
        response = self.client.post(
            self.create_api_key_url,
            json.dumps({"group_name": "test_group"}),
            content_type="application/json",
        )
        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)
        self.assertEqual(
            response.data, {"error": "API key already exists for this group"}
        )


class TokenViewTests(APITestCase):
    def setUp(self):
        self.user = User.objects.create_user(
            username="testuser", password="testpass", email="testuser@orange.com"
        )
        self.url = reverse("token_obtain")

        # setUp patchs
        self.api_key_patcher = patch("api.views.APIKey.objects.get")
        self.create_api_key_patcher = patch("api.views.create_api_key")

        # start mocks
        self.mock_api_key_get = self.api_key_patcher.start()
        self.mock_create_api_key = self.create_api_key_patcher.start()

        # mocks config
        self.mock_api_key = Mock()
        self.mock_api_key.is_valid.return_value = True
        self.mock_api_key_get.return_value = self.mock_api_key
        self.mock_create_api_key.return_value = {"api_key": "mocked_api_key"}

        self.headers = {"Authorization": self.mock_create_api_key().get("api_key")}

    def tearDown(self):
        self.api_key_patcher.stop()
        self.create_api_key_patcher.stop()

    def test_token_generation(self):
        response = self.client.post(
            self.url,
            json.dumps({"cuid": "abcd1234", "email": "testuser@orange.com"}),
            headers=self.headers,
            content_type="application/json",
        )
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertIn("access_token", response.data)

    def test_token_generation_with_invalid_cuid(self):
        response = self.client.post(
            self.url,
            json.dumps({"cuid": "invalid_cuid", "email": "testuser@orange.com"}),
            headers=self.headers,
            content_type="application/json",
        )
        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)
        self.assertIn("cuid", response.data)

    def test_token_generation_without_email(self):
        response = self.client.post(
            self.url,
            json.dumps({"cuid": "abcd1234"}),
            headers=self.headers,
            content_type="application/json",
        )
        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)
        self.assertEqual(
            response.data, {"error": "Email required for new user registration"}
        )


class CreateTaskViewTests(APITestCase):
    def setup_mocks(self):
        # setUp patchs
        self.api_key_patcher = patch("api.views.APIKey.objects.get")
        self.create_api_key_patcher = patch("api.views.create_api_key")
        self.access_token_patcher = patch("api.views.AccessToken.for_user")
        self.token_validator_patcher = patch(
            "rest_framework_simplejwt.authentication.JWTAuthentication.get_validated_token"
        )
        self.get_queue_patcher = patch("api.views.get_queue")
        self.job_queue_rank_patcher = patch("api.views.job_queue_rank")

        # start mocks
        self.mock_api_key_get = self.api_key_patcher.start()
        self.mock_create_api_key = self.create_api_key_patcher.start()
        self.mock_access_token = self.access_token_patcher.start()
        self.mock_token_validator = self.token_validator_patcher.start()
        self.mock_get_queue = self.get_queue_patcher.start()
        self.mock_job_queue_rank = self.job_queue_rank_patcher.start()

        # mocks config
        self.mock_api_key = Mock()
        self.mock_api_key.is_valid.return_value = True
        self.mock_api_key_get.return_value = self.mock_api_key
        self.mock_create_api_key.return_value = {"api_key": "mocked_api_key"}
        headers = {"Authorization": self.mock_create_api_key().get("api_key")}

        self.mock_token = Mock()
        self.mock_token.set_exp.return_value = None
        self.mock_token.__str__ = Mock(return_value="mocked.access.token")
        self.mock_access_token.return_value = self.mock_token

        validated_token = Mock()
        validated_token.__getitem__ = Mock(
            side_effect=lambda x: self.user.id if x == "user_id" else None
        )
        validated_token.get = Mock(
            side_effect=lambda x, default=None: (
                self.user.id if x == "user_id" else default
            )
        )
        self.mock_token_validator.return_value = validated_token
        self.mock_queue = MagicMock()
        self.mock_job = MagicMock()
        self.mock_job.id = "mocked_job_id"
        self.mock_job.created_at = datetime.now().isoformat()
        self.mock_job.get_status.return_value = "queued"

        self.mock_get_queue.return_value = self.mock_queue
        self.mock_queue.enqueue.return_value = self.mock_job
        self.mock_job_queue_rank.return_value = 1
        return headers

    def obtain_token(self):
        return self.client.post(
            reverse("token_obtain"),
            json.dumps({"cuid": "abcd1234", "email": "testuser@orange.com"}),
            content_type="application/json",
            headers=self.headers,
        )

    def setUp(self):
        self.user = User.objects.create_user(
            username="testuser", password="testpass", email="testuser@orange.com"
        )
        self.url = reverse("create_batch")
        self.valid_payload = json.dumps(
            {
                "cuid": "abcd1234",
                "scope": {
                    "zone": "eqt",
                    "ic01_list": ["ic01_1"],
                    "period_start": "2024-01-01",
                    "period_end": "2024-09-30",
                },
                "prompts": [{"name": "prompt_1", "value": "prompt_1_value"}],
                "fields": ["typeLabel"],
            }
        )
        self.headers = self.setup_mocks()
        self.token_response = self.obtain_token()

        self.assertEqual(self.token_response.status_code, status.HTTP_200_OK)
        self.assertEqual(
            self.token_response.data.get("access_token"), "mocked.access.token"
        )

    def tearDown(self):
        self.api_key_patcher.stop()
        self.access_token_patcher.stop()
        self.token_validator_patcher.stop()
        self.get_queue_patcher.stop()
        self.job_queue_rank_patcher.stop()

    def test_create_batch_task(self):
        response = self.client.post(
            self.url,
            self.valid_payload,
            content_type="application/json",
            headers={
                "Authorization": f"Bearer {self.token_response.data.get('access_token')}"
            },
        )

        self.assertEqual(response.status_code, status.HTTP_202_ACCEPTED)
        self.assertIn("job_id", response.data)
        self.assertEqual(response.data["job_id"], "mocked_job_id")
        self.assertEqual(response.data["status"], "queued")
        self.assertEqual(response.data["queue_rank"], 1)

    def test_create_task_unauthenticated(self):
        self.client.logout()
        response = self.client.post(self.url)
        self.assertEqual(response.status_code, status.HTTP_401_UNAUTHORIZED)

    def test_create_batch_task_without_cuid(self):
        response = self.client.post(
            self.url,
            json.dumps(
                {
                    "scope": {
                        "ic01_list": ["ic01_1"],
                        "period_start": "2024-01-01",
                        "period_end": "2024-09-30",
                    },
                    "prompts": [{"name": "prompt_1", "value": "prompt_1_value"}],
                    "fields": ["typeLabel"],
                }
            ),
            content_type="application/json",
            headers={
                "Authorization": f"Bearer {self.token_response.data.get('access_token')}"
            },
        )
        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)
        self.assertIn("cuid", response.data)

    def test_create_batch_task_with_invalid_dates(self):
        response = self.client.post(
            self.url,
            json.dumps(
                {
                    "cuid": "abcd1234",
                    "scope": {
                        "zone": "eqt",
                        "ic01_list": ["ic01_1"],
                        "period_start": "2024-09-30",
                        "period_end": "2024-01-01",  # Invalid: start date is after end date
                    },
                    "prompts": [{"name": "prompt_1", "value": "prompt_1_value"}],
                    "fields": ["typeLabel"],
                }
            ),
            content_type="application/json",
            headers={
                "Authorization": f"Bearer {self.token_response.data.get('access_token')}"
            },
        )
        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)
        self.assertIn(
            "period_start", str(response.data.get("scope").get("non_field_errors")[0])
        )

    def test_create_batch_task_with_filters(self):
        response = self.client.post(
            self.url,
            json.dumps(
                {
                    "cuid": "abcd1234",
                    "scope": {
                        "zone": "eqt",
                        "ic01_list": ["ic01_1"],
                        "period_start": "2024-01-01",
                        "period_end": "2024-09-30",
                        "filters": {
                            "status": "SGDG,PEND",
                            "contract_type": "CPSE",
                            "sales_country": {"value": ["country1", "country2"]},
                            "sales_region": {"value": ["region1", "region2"]},
                            "document_label": {
                                "value": ["Legal"],
                                "filter_type": "keep",
                            },
                            "original_file_name": {"value": ["awt"]},
                            "document_creation_date": {
                                "start_period": "2024-06-27",
                                "end_period": "2024-06-28",
                            },
                        },
                    },
                    "prompts": [{"name": "prompt_1", "value": "prompt_1_value"}],
                    "fields": ["typeLabel"],
                }
            ),
            content_type="application/json",
            headers={
                "Authorization": f"Bearer {self.token_response.data.get('access_token')}"
            },
        )

        self.assertEqual(response.status_code, status.HTTP_202_ACCEPTED)
        self.assertIn("job_id", response.data)

    def test_create_batch_task_with_invalid_filters(self):
        response = self.client.post(
            self.url,
            json.dumps(
                {
                    "cuid": "abcd1234",
                    "scope": {
                        "zone": "eqt",
                        "ic01_list": ["ic01_1"],
                        "period_start": "2024-01-01",
                        "period_end": "2024-09-30",
                        "filters": {
                            "status": "SGDG,PEND",
                            "contract_type": "CPSE",
                            "sales_country": {"value": ["country1", "country2"]},
                            "sales_region": {"value": ["region1", "region2"]},
                            "document_label": {
                                "value": ["Legal"],
                                "filter_type": "invalid_filter_type",
                            },  # Invalid filter type
                        },
                    },
                    "prompts": [{"name": "prompt_1", "value": "prompt_1_value"}],
                    "fields": ["typeLabel"],
                }
            ),
            content_type="application/json",
            headers={
                "Authorization": f"Bearer {self.token_response.data.get('access_token')}"
            },
        )

        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)
        self.assertIn(
            "invalid_choice",
            response.data.get("scope")
            .get("filters")
            .get("document_label")
            .get("filter_type")[0]
            .code,
        )

    def test_create_batch_task_with_missing_filter_values(self):
        response = self.client.post(
            self.url,
            json.dumps(
                {
                    "cuid": "abcd1234",
                    "scope": {
                        "zone": "eqt",
                        "ic01_list": ["ic01_1"],
                        "period_start": "2024-01-01",
                        "period_end": "2024-09-30",
                        "filters": {
                            "status": "SGDG,PEND",
                            "contract_type": "CPSE",
                            "sales_country": {"value": []},  # Missing values
                        },
                    },
                    "prompts": [{"name": "prompt_1", "value": "prompt_1_value"}],
                    "fields": ["typeLabel"],
                }
            ),
            content_type="application/json",
            headers={
                "Authorization": f"Bearer {self.token_response.data.get('access_token')}"
            },
        )

        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)
        self.assertIn(
            "list can not be empty",
            str(
                response.data.get("scope")
                .get("filters")
                .get("sales_country")
                .get("value")[0]
            ),
        )

    def test_create_batch_task_without_zone(self):
        response = self.client.post(
            self.url,
            json.dumps(
                {
                    "cuid": "abcd1234",
                    "scope": {
                        "ic01_list": ["ic01_1"],
                        "period_start": "2024-01-01",
                        "period_end": "2024-09-30",
                        "filters": {
                            "status": "SGDG,PEND",
                            "contract_type": "CPSE",
                            "sales_country": {"value": ["country1", "country2"]},
                        },
                    },
                    "prompts": [{"name": "prompt_1", "value": "prompt_1_value"}],
                    "fields": ["typeLabel"],
                }
            ),
            content_type="application/json",
            headers={
                "Authorization": f"Bearer {self.token_response.data.get('access_token')}"
            },
        )

        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)
        self.assertIn(
            "This field is required",
            str(response.data.get("scope").get("zone")),
        )

    def test_create_batch_task_with_invalid_zone(self):
        response = self.client.post(
            self.url,
            json.dumps(
                {
                    "cuid": "abcd1234",
                    "scope": {
                        "zone": "invalid_zone",  # invalid zone
                        "ic01_list": ["ic01_1"],
                        "period_start": "2024-01-01",
                        "period_end": "2024-09-30",
                        "filters": {
                            "status": "SGDG,PEND",
                            "contract_type": "CPSE",
                            "sales_country": {"value": ["country1", "country2"]},
                        },
                    },
                    "prompts": [{"name": "prompt_1", "value": "prompt_1_value"}],
                    "fields": ["typeLabel"],
                }
            ),
            content_type="application/json",
            headers={
                "Authorization": f"Bearer {self.token_response.data.get('access_token')}"
            },
        )

        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)
        self.assertIn("invalid", response.data.get("scope").get("zone")[0].code)

    def test_create_batch_task_with_valid_prompts_dependencies(self):
        payload = json.loads(self.valid_payload)
        payload["prompts"] = [
            {"name": "prompt_1", "value": "val1"},
            {"name": "prompt_2", "value": "val2", "dependencies": ["prompt_1"]},
        ]
        response = self.client.post(
            self.url,
            json.dumps(payload),
            content_type="application/json",
            headers={
                "Authorization": f"Bearer {self.token_response.data.get('access_token')}"
            },
        )
        self.assertEqual(response.status_code, status.HTTP_202_ACCEPTED)

    def test_create_batch_task_with_nonexistent_prompt_dependency(self):
        payload = json.loads(self.valid_payload)
        payload["prompts"] = [
            {"name": "prompt_1", "value": "val1"},
            {
                "name": "prompt_2",
                "value": "val2",
                "dependencies": ["nonexistent_prompt"],
            },
        ]
        response = self.client.post(
            self.url,
            json.dumps(payload),
            content_type="application/json",
            headers={
                "Authorization": f"Bearer {self.token_response.data.get('access_token')}"
            },
        )
        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)
        self.assertIn("does not reference an existing prompt", str(response.data))

    def test_create_batch_task_with_empty_dependencies_list(self):
        payload = json.loads(self.valid_payload)
        payload["prompts"] = [
            {"name": "prompt_1", "value": "val1", "dependencies": []},
        ]
        response = self.client.post(
            self.url,
            json.dumps(payload),
            content_type="application/json",
            headers={
                "Authorization": f"Bearer {self.token_response.data.get('access_token')}"
            },
        )
        self.assertEqual(response.status_code, status.HTTP_202_ACCEPTED)

    def test_create_batch_task_with_missing_dependencies_field(self):
        payload = json.loads(self.valid_payload)
        payload["prompts"] = [
            {"name": "prompt_1", "value": "val1"},
        ]
        response = self.client.post(
            self.url,
            json.dumps(payload),
            content_type="application/json",
            headers={
                "Authorization": f"Bearer {self.token_response.data.get('access_token')}"
            },
        )
        self.assertEqual(response.status_code, status.HTTP_202_ACCEPTED)

    def test_create_batch_task_with_duplicate_prompt_names_in_dependencies(self):
        payload = json.loads(self.valid_payload)
        payload["prompts"] = [
            {"name": "prompt_1", "value": "val1"},
            {
                "name": "prompt_2",
                "value": "val2",
                "dependencies": ["prompt_1", "prompt_1"],
            },
        ]
        response = self.client.post(
            self.url,
            json.dumps(payload),
            content_type="application/json",
            headers={
                "Authorization": f"Bearer {self.token_response.data.get('access_token')}"
            },
        )
        self.assertEqual(response.status_code, status.HTTP_202_ACCEPTED)

    def test_create_batch_task_with_invalid_type_in_dependencies(self):
        payload = json.loads(self.valid_payload)
        payload["prompts"] = [
            {"name": "prompt_1", "value": "val1"},
            {"name": "prompt_2", "value": "val2", "dependencies": "prompt_1"},
        ]
        response = self.client.post(
            self.url,
            json.dumps(payload),
            content_type="application/json",
            headers={
                "Authorization": f"Bearer {self.token_response.data.get('access_token')}"
            },
        )
        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)
        self.assertIn(
            "not_a_list", response.data.get("prompts")[1].get("dependencies")[0].code
        )


class GetDeleteTaskViewTests(CreateTaskViewTests):
    def setUp(self):
        super().setUp()
        self.create_response = self.client.post(
            reverse("create_batch"),
            self.valid_payload,
            content_type="application/json",
            headers={
                "Authorization": f"Bearer {self.token_response.data.get('access_token')}"
            },
        )
        self.job_id = self.create_response.data["job_id"]
        self.job_fetch_patcher = patch("api.views.Job.fetch")
        self.mock_job_fetch = self.job_fetch_patcher.start()
        self.mock_job_fetch.return_value = self.mock_job

    def tearDown(self):
        super().tearDown()
        self.job_fetch_patcher.stop()

    def test_get_task(self):
        url = reverse("get_delete_batch", args=[self.job_id])
        response = self.client.get(
            url,
            headers={
                "Authorization": f"Bearer {self.token_response.data.get('access_token')}"
            },
        )
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(response.data["status"], "queued")
        self.assertEqual(response.data["queue_rank"], 1)

    def test_get_task_with_invalid_job_id(self):
        self.mock_job_fetch.side_effect = NoSuchJobError()
        url = reverse("get_delete_batch", args=["invalid_job_id"])
        response = self.client.get(
            url,
            headers={
                "Authorization": f"Bearer {self.token_response.data.get('access_token')}"
            },
        )
        self.assertEqual(response.status_code, status.HTTP_404_NOT_FOUND)
        self.assertIn("message", response.data)

    def test_delete_task(self):
        with patch("api.views.send_stop_job_command") as _:
            with patch("api.views.time.sleep"):
                url = reverse("get_delete_batch", args=[self.job_id])
                response = self.client.delete(
                    url,
                    headers={
                        "Authorization": f"Bearer {self.token_response.data.get('access_token')}"
                    },
                )

                self.assertEqual(response.status_code, status.HTTP_204_NO_CONTENT)
                self.assertEqual({}, response.data)

    def test_delete_task_with_invalid_job_id(self):
        self.mock_job_fetch.side_effect = NoSuchJobError()
        url = reverse("get_delete_batch", args=["invalid_job_id"])
        response = self.client.delete(
            url,
            headers={
                "Authorization": f"Bearer {self.token_response.data.get('access_token')}"
            },
        )
        self.assertEqual(response.status_code, status.HTTP_404_NOT_FOUND)
        self.assertIn("message", response.data)


class TaskResultViewTests(CreateTaskViewTests):
    def setUp(self):
        super().setUp()

        self.job_fetch_patcher = patch("api.views.Job.fetch")
        self.mock_job_fetch = self.job_fetch_patcher.start()
        self.mock_job_fetch.return_value = self.mock_job

        self.os_path_exists_patcher = patch("api.views.os.path.exists")
        self.mock_os_path_exists = self.os_path_exists_patcher.start()
        self.mock_os_path_exists.return_value = True

        self.open_patcher = patch("builtins.open", create=True)
        self.mock_open = self.open_patcher.start()
        self.mock_file = MagicMock()
        self.mock_open.return_value.__enter__.return_value = self.mock_file
        self.mock_file.read.return_value = b"Test content"

        self.b64encode_patcher = patch("api.views.base64.b64encode")
        self.mock_b64encode = self.b64encode_patcher.start()
        self.mock_b64encode.return_value.decode.return_value = "base64_encoded_content"

    def tearDown(self):
        super().tearDown()
        self.job_fetch_patcher.stop()
        self.os_path_exists_patcher.stop()
        self.open_patcher.stop()
        self.b64encode_patcher.stop()

    def test_get_task_result(self):
        response = self.client.get(
            reverse("get_batch_result", args=[self.mock_job.id]),
            headers={
                "Authorization": f"Bearer {self.token_response.data.get('access_token')}"
            },
        )
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertIn("file_content", response.data)
        self.assertEqual(response.data["file_content"], "base64_encoded_content")
        self.assertEqual(response.data["file_name"], f"{self.mock_job.id}.xlsx")

        self.mock_job_fetch.assert_called_once()
        self.mock_open.assert_called_once()
        self.mock_file.read.assert_called_once()
        self.mock_b64encode.assert_called_once_with(b"Test content")

    def test_get_task_result_file_not_found(self):
        self.mock_os_path_exists.return_value = False

        response = self.client.get(
            reverse("get_batch_result", args=[self.mock_job.id]),
            headers={
                "Authorization": f"Bearer {self.token_response.data.get('access_token')}"
            },
        )
        self.assertEqual(response.status_code, status.HTTP_404_NOT_FOUND)
        self.assertIn("error", response.data)

    def test_get_task_result_with_invalid_job_id(self):
        self.mock_job_fetch.side_effect = NoSuchJobError()

        response = self.client.get(
            reverse("get_batch_result", args=["invalid_job_id"]),
            headers={
                "Authorization": f"Bearer {self.token_response.data.get('access_token')}"
            },
        )
        self.assertEqual(response.status_code, status.HTTP_404_NOT_FOUND)
        self.assertIn("message", response.data)
