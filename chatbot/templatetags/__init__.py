from django import template

register = template.Library()

# Import your custom filters
from .custom_filters import filesizeformat_mb

# Register your custom filters
register.filter('filesizeformat_mb', filesizeformat_mb)
