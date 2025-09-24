from django.contrib import admin
from django.contrib.auth.models import Group

from api.models import APIKey


class APIKeyAdmin(admin.ModelAdmin):
    list_display = ("group", "key", "created_at", "expires_at", "is_revoked")
    search_fields = ("group__name", "key")
    list_filter = ("is_revoked", "expires_at")


class UserInline(admin.TabularInline):
    model = Group.user_set.through
    extra = 0


class GroupAdmin(admin.ModelAdmin):
    inlines = [UserInline]


admin.site.register(APIKey, APIKeyAdmin)
admin.site.unregister(Group)
admin.site.register(Group, GroupAdmin)
