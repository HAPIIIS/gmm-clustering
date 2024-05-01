from django import template

register = template.Library()

@register.filter(name='remove_quotes')
def remove_quotes(value):
  """Removes leading and trailing double quotes."""
  if value and value[0] == '"' and value[-1] == '"':
    return value[1:-1]  # Remove quotes if they exist
  else:
    return value

