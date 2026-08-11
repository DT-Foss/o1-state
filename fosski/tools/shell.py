"""Execute shell commands safely."""

description = "Run a shell command and return the output"
keywords = ['run', 'execute', 'shell', 'bash', 'command']
patterns = [
    r'(?:run|execute)\s+[`"\']?(.+?)[`"\']?\s*$',
    r'^(ls|pwd|git\s|pip\s|python3?\s|node\s|make\s|wc\s|grep\s|find\s|head\s|tail\s)\s*(.*)$',
]

import subprocess

BLOCKED = {'rm -rf /', 'rm -rf ~', 'sudo rm', 'mkfs', 'dd if=', ':(){:|:&};:'}

def execute(args):
    """Run a shell command."""
    command = args.get('command', '')
    timeout = args.get('timeout', 30)
    cwd = args.get('cwd', '.')

    if not command:
        return {'success': False, 'response': 'No command provided.'}

    # Security
    for blocked in BLOCKED:
        if blocked in command.lower():
            return {'success': False, 'response': f'Blocked: {command}'}

    try:
        result = subprocess.run(
            command, shell=True, capture_output=True,
            text=True, timeout=timeout, cwd=cwd,
        )
        output = result.stdout
        if result.stderr:
            output += f"\nSTDERR:\n{result.stderr}"
        return {
            'success': result.returncode == 0,
            'stdout': result.stdout,
            'stderr': result.stderr,
            'returncode': result.returncode,
            'response': f"```\n$ {command}\n{output.strip()}\n```",
        }
    except subprocess.TimeoutExpired:
        return {'success': False, 'response': f'Timed out after {timeout}s'}
    except Exception as e:
        return {'success': False, 'response': f'Error: {e}'}
