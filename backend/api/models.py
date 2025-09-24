import secrets

from django.contrib.auth.models import Group, User
from django.db import models
from django.utils import timezone


class APIKey(models.Model):
    group = models.OneToOneField(Group, on_delete=models.CASCADE)
    key = models.CharField(max_length=60, unique=True)
    created_at = models.DateTimeField(auto_now_add=True)
    expires_at = models.DateTimeField(null=True, blank=True)
    is_revoked = models.BooleanField(default=False)

    @classmethod
    def generate_key(cls):
        return secrets.token_hex(30)

    def is_valid(self) -> bool:
        """Check if the api key is valid

        Returns:
            bool: True if the key is valid, False otherwise
        """
        return (
            not (self.expires_at and self.expires_at < timezone.now())
            and not self.is_revoked
        )


class UserContract(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    ic01 = models.CharField(max_length=20)
    file_name = models.CharField()
    source = models.CharField()
    contract_amendment_id = models.CharField()
    contract_type = models.CharField()
    contract_type_label = models.CharField()
    document_id = models.IntegerField()
    creation_date = models.DateTimeField()
    indexation_date = models.DateTimeField()
    chunk_id_first = models.CharField(null=True, blank=True)
    chunk_id_last = models.CharField(null=True, blank=True)

    class Meta:
        unique_together = (
            (
                "user",
                "ic01",
                "source",
                "document_id",
                "creation_date",
            ),
        )
