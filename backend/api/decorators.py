import ipaddress
import logging

from rest_framework import status
from rest_framework.response import Response

from api.models import APIKey

logger = logging.getLogger(__name__)


def api_key_required(view_func):
    def _wrapped_view(request, *args, **kwargs):
        api_key = request.request.headers.get("Authorization")
        if not api_key:
            return Response(
                {"error": "API Key missing"}, status=status.HTTP_401_UNAUTHORIZED
            )
        try:
            api_key_obj = APIKey.objects.get(key=api_key)
            if not api_key_obj.is_valid():
                return Response(
                    {"error": "API Key expired or revoked"},
                    status=status.HTTP_401_UNAUTHORIZED,
                )
        except APIKey.DoesNotExist:
            return Response(
                {"error": "Invalid API Key"}, status=status.HTTP_401_UNAUTHORIZED
            )
        request.api_key = api_key_obj
        return view_func(request, *args, **kwargs)

    return _wrapped_view


def allowed_kube_probe(view_func):
    def _wrapped_view(request, *args, **kwargs):
        try:
            client_ip = request.META.get("REMOTE_ADDR", "")
            ip_obj = ipaddress.ipaddress(client_ip)
            is_internal = ip_obj in ipaddress.ip_network("192.0.0.0/16")
            is_probe = request.headers.get("User-Agent", "").startswith("kube-probe")
            if is_internal and is_probe:
                return view_func(request, *args, **kwargs)
        except ValueError as e:
            return Response(
                {"error": f"Invalid IP format. {str(e)}"},
                status=status.HTTP_400_BAD_REQUEST,
            )

    return _wrapped_view
