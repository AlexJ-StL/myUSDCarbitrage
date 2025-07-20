"""Custom Jinja2 filters for report templates."""

from datetime import datetime


def setup_jinja_filters(env):
    """Set up custom Jinja2 filters for the environment.

    Args:
        env: Jinja2 environment to add filters to
    """
    # Add a filter to get the current year
    env.filters["now"] = now_filter


def now_filter(format_string):
    """Return the current date/time formatted according to the format string.

    Args:
        format_string: Format string (e.g., 'Y' for year)

    Returns:
        str: Formatted date/time string
    """
    return datetime.now().strftime(format_string.replace("Y", "%Y"))
