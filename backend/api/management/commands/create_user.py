from django.conf import settings
from django.contrib.auth.models import Group, User
from django.core.management.base import BaseCommand

from utils.utils import generate_random_password, is_valid_cuid


class Command(BaseCommand):
    help = "Create superuser if it does not exist"

    def handle(self, *args, **options):
        filestalk_group_name = "filestalk"
        self.create_group(filestalk_group_name)
        self.create_superusers(settings.DEV_USERS, filestalk_group_name)
        self.create_groups(settings.USER_PARTNERS)

    def create_group(self, group_name):
        group, created = Group.objects.get_or_create(name=group_name)
        if created:
            self.stdout.write(
                self.style.SUCCESS(f"Group {group_name} created successfully")
            )

    def create_superusers(self, dev_users, group_name):
        group = Group.objects.get(name=group_name)
        for dev_user in dev_users:
            username = dev_user.get("username")
            if (
                is_valid_cuid(username)
                and not User.objects.filter(username=username).exists()
            ):
                user = User.objects.create_superuser(
                    username=username,
                    password=generate_random_password(),
                    email=dev_user.get("email"),
                )
                self.stdout.write(
                    self.style.SUCCESS(f"Superuser {username} created successfully")
                )
                group.user_set.add(user)
                self.stdout.write(
                    self.style.SUCCESS(
                        f"Superuser {username} added to {group_name} group"
                    )
                )

    def create_groups(self, group_names):
        for group_name in group_names:
            self.create_group(group_name)
