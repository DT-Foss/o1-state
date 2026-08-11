"""Write content to a file."""

description = "Write or create a file with given content"
keywords = ['write', 'save', 'create file', 'write to']
patterns = [
    r'(?:write|save)\s+(?:to\s+)?(?:file\s+)?(\S+)',
    r'(?:create)\s+(?:a\s+)?(?:file\s+)?(\S+)',
]

import os

def execute(args):
    """Write content to a file."""
    path = args.get('path', '')
    content = args.get('content', '')

    if not path:
        return {'success': False, 'response': 'No file path provided.'}
    if not content:
        return {'success': False, 'response': 'No content provided.'}

    try:
        os.makedirs(os.path.dirname(os.path.abspath(path)) or '.', exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            f.write(content)
        return {
            'success': True,
            'response': f"Written {len(content)} characters to `{path}`.",
        }
    except Exception as e:
        return {'success': False, 'response': f"Error writing {path}: {e}"}
