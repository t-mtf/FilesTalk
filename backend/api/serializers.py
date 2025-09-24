from datetime import date

from rest_framework import serializers

from config.settings import CONTRACT_FIELDS


class TokenSerializer(serializers.Serializer):
    """Serializer for generating access token."""

    cuid = serializers.CharField(required=True)
    email = serializers.EmailField(required=False)

    def validate_cuid(self, value):
        """Validate the cuid format."""
        if len(value) != 8 or not value[:4].isalpha() or not value[4:].isdigit():
            raise serializers.ValidationError("Cuid not valid.")
        return value

    def validate_email(self, value):
        """Validate the email domain."""
        if value and not value.endswith("@orange.com"):
            raise serializers.ValidationError(
                "Email must be from the domain '@orange.com'."
            )
        return value


class FilterValueSerializer(serializers.Serializer):
    """Serializer for filter values."""

    value = serializers.ListField(child=serializers.CharField(), required=True)
    filter_type = serializers.ChoiceField(choices=["keep", "remove"], required=False)

    def validate_value(self, value):
        """Validate that value list is not empty"""
        if not value:
            raise serializers.ValidationError("Value list can not be empty.")
        return value


class DocumentCreationDateSerializer(serializers.Serializer):
    """Serializer for document creation date filters."""

    start_period = serializers.DateField(format="%Y-%m-%d", required=True)
    end_period = serializers.DateField(format="%Y-%m-%d", required=True)

    def validate(self, attrs):
        start_period = attrs.get("start_period")
        end_period = attrs.get("end_period")
        if start_period and end_period and start_period > end_period:
            raise serializers.ValidationError(
                "The start_period date cannot be later than the end_period date."
            )
        return attrs

    def validate_end_period(self, value):
        """Validate that the end_period date is not in the future."""
        if value > date.today():
            raise serializers.ValidationError(
                "The end_period date cannot be in the future."
            )
        return value


class FiltersSerializer(serializers.Serializer):
    """Serializer for the filters field."""

    status = serializers.CharField(required=False)
    contract_type = serializers.CharField(required=False)
    document_label = FilterValueSerializer(required=False)
    sales_country = FilterValueSerializer(required=False)
    sales_region = FilterValueSerializer(required=False)
    original_file_name = FilterValueSerializer(required=False)
    document_creation_date = DocumentCreationDateSerializer(required=False)


class ScopeSerializer(serializers.Serializer):
    """Serializer for the scope of the batch processing task."""

    zone = serializers.CharField(required=True)
    ic01_list = serializers.ListField(child=serializers.CharField(), required=False)
    period_start = serializers.DateField(required=False)
    period_end = serializers.DateField(required=False)
    filters = FiltersSerializer(required=False)

    # added for test
    id_list = serializers.ListField(child=serializers.CharField(), required=False)

    def validate(self, attrs):
        """Validate that at least one of the required fields is present."""
        ic01_list = attrs.get("ic01_list")
        period_start = attrs.get("period_start")
        period_end = attrs.get("period_end")
        # added for test
        id_list = attrs.get("id_list")

        # add for test
        if not (ic01_list or id_list) and (not period_start or not period_end):
            raise serializers.ValidationError(
                "At least ic01_list or id_list or both period_start and period_end must be defined."
            )
        if (period_start or period_end) and (not period_start or not period_end):
            raise serializers.ValidationError(
                "Both period_start and period_end must be defined."
            )
        if period_start and period_end and period_start > period_end:
            raise serializers.ValidationError(
                "The period_start date cannot be later than the period_end date."
            )
        return attrs

    def validate_zone(self, value):
        """Validate that the zone is either 'eqt' or 'ftsa' (case insensitive)."""
        valid_zones = ["eqt", "ftsa"]
        if value.lower() not in valid_zones:
            raise serializers.ValidationError(
                "The zone must be either 'eqt' or 'ftsa'."
            )
        return value

    def validate_period_end(self, value):
        """Validate that the period_end date is not in the future."""
        if value > date.today():
            raise serializers.ValidationError(
                "The period_end date cannot be in the future."
            )
        return value


class PromptSerializer(serializers.Serializer):
    """Serializer for individual prompts."""

    name = serializers.CharField()
    value = serializers.CharField()
    dependencies = serializers.ListField(child=serializers.CharField(), required=False)


class CreateTaskSerializer(serializers.Serializer):
    """Serializer for creating a new batch processing task."""

    cuid = serializers.CharField(required=True)
    scope = ScopeSerializer(required=True)
    prompts = PromptSerializer(required=True, many=True)
    fields = serializers.ListField(
        child=serializers.ChoiceField(choices=CONTRACT_FIELDS), required=False
    )

    def validate_prompts(self, value):
        """Validate that the prompts list is not empty."""
        if not value:
            raise serializers.ValidationError("The prompts list cannot be empty.")

        prompt_names: set = {prompt.get("name") for prompt in value}

        for prompt in value:
            if "dependencies" in prompt and prompt.get("dependencies"):
                for dependency in prompt.get("dependencies"):
                    if dependency not in prompt_names:
                        raise serializers.ValidationError(
                            f"Dependency {dependency} in prompt {prompt.get('name')} does not reference an existing prompt"
                        )
        return value
