from django import template

register = template.Library()

@register.filter
def filesizeformat_mb(value):
    """
    Converts a file size from bytes to megabytes (MB).
    """
    try:
        value = float(value)
    except (TypeError, ValueError):
        return "0 MB"

    mb_size = value / (1024 * 1024)
    return "%.2f MB" % mb_size
