from django.contrib import admin
from django.urls import path, re_path
from drf_yasg import openapi
from drf_yasg.views import get_schema_view
from rest_framework import permissions

from .views import (
    CreateTaskView,
    GetDeleteTaskView,
    HelloWorldView,
    TaskResultView,
    TokenView,
    create_api_key,
    revoke_api_key,
)

schema_view = get_schema_view(
    openapi.Info(
        title="Files Talk API",
        default_version="v1",
        description="Documentation of API endpoints",
        terms_of_service="https://www.example.com/policies/terms/",
        contact=openapi.Contact(email="filestalk@orange.com"),
        license=openapi.License(name="BSD License"),
    ),
    public=True,
    permission_classes=(permissions.AllowAny,),
)

urlpatterns = [
    re_path(r"^hello/?$", HelloWorldView.as_view(), name="hello world"),
    re_path(r"^create-api-key/?$", create_api_key, name="create_api_key"),
    re_path(r"^revoke-api-key/?$", revoke_api_key, name="revoke_api_key"),
    re_path(r"^token/?$", TokenView.as_view(), name="token_obtain"),
    re_path(r"^batch/?$", CreateTaskView.as_view(), name="create_batch"),
    re_path(
        r"^batch/(?P<job_id>[^/]+)/?$",
        GetDeleteTaskView.as_view(),
        name="get_delete_batch",
    ),
    re_path(
        r"^batch/(?P<job_id>[^/]+)/file/?$",
        TaskResultView.as_view(),
        name="get_batch_result",
    ),
    re_path(
        r"^swagger(?P<format>\.json|\.yaml)$",
        schema_view.without_ui(cache_timeout=0),
        name="schema-json",
    ),
    path(
        "swagger/",
        schema_view.with_ui("swagger", cache_timeout=0),
        name="schema-swagger-ui",
    ),
    path("redoc/", schema_view.with_ui("redoc", cache_timeout=0), name="schema-redoc"),
    path("admin/", admin.site.urls),
]
