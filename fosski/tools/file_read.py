"""Read file contents."""

description = "Read the contents of a file"
keywords = ['read', 'open', 'show', 'cat', 'display', 'view']
patterns = [
    r'(?:read|open|show|cat|display|view)\s+(?:the\s+)?(?:file\s+)?(?:contents?\s+of\s+)?(\S+)',
]

def execute(args):
    """Read a file and return its contents."""
    path = args.get('path', '')
    if not path:
        return {'success': False, 'response': 'No file path provided.'}

    try:
        with open(path, 'r', encoding='utf-8') as f:
            content = f.read()
        lines = content.split('\n')
        # Show with line numbers
        numbered = '\n'.join(f"{i+1:4d} | {line}" for i, line in enumerate(lines))
        return {
            'success': True,
            'content': content,
            'lines': len(lines),
            'response': f"File `{path}` ({len(lines)} lines):\n```\n{numbered}\n```",
        }
    except FileNotFoundError:
        return {'success': False, 'response': f"File not found: {path}"}
    except Exception as e:
        return {'success': False, 'response': f"Error reading {path}: {e}"}
