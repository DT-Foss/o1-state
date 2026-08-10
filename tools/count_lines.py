"""Count lines in a file."""

description = "Count the number of lines in a file"
keywords = ['count', 'lines', 'wc', 'how many lines']
patterns = [
    r'(?:count|how many)\s+(?:the\s+)?lines?\s+(?:in\s+)?(\S+)',
    r'wc\s+-?l\s+(\S+)',
]

def execute(args):
    """Count lines in a file."""
    path = args.get('path', '')
    if not path:
        return {'success': False, 'response': 'No file path provided.'}

    try:
        with open(path, 'r', encoding='utf-8') as f:
            lines = sum(1 for _ in f)
        return {
            'success': True,
            'result': lines,
            'response': f"`{path}` has {lines} lines.",
        }
    except FileNotFoundError:
        return {'success': False, 'response': f"File not found: {path}"}
    except Exception as e:
        return {'success': False, 'response': f"Error: {e}"}
