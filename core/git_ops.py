"""
Git Operations — Local Git Integration
=========================================
Provides git status, diff, log, and commit operations
for FOSS-KI's tool system without external git libraries.

Uses subprocess to call git directly.
"""

import subprocess
import os
import re
from typing import Dict, Any, List, Tuple


def _run_git(args: List[str], cwd: str = '.') -> Tuple[bool, str]:
    """Run a git command and return (success, output)."""
    try:
        result = subprocess.run(
            ['git'] + args,
            capture_output=True, text=True,
            cwd=cwd, timeout=30
        )
        if result.returncode == 0:
            return True, result.stdout.rstrip('\n')
        return False, result.stderr.strip()
    except FileNotFoundError:
        return False, "git not found"
    except subprocess.TimeoutExpired:
        return False, "git command timed out"


class GitOps:
    """Git operations for a repository."""

    def __init__(self, repo_path: str = '.'):
        self.repo_path = os.path.abspath(repo_path)

    def is_repo(self) -> bool:
        """Check if the path is a git repository."""
        ok, _ = _run_git(['rev-parse', '--is-inside-work-tree'], self.repo_path)
        return ok

    def status(self) -> Dict[str, Any]:
        """Get git status as structured data."""
        ok, output = _run_git(['status', '--porcelain'], self.repo_path)
        if not ok:
            return {'error': output}

        staged = []
        modified = []
        untracked = []

        for line in output.split('\n'):
            if not line:
                continue
            x, y = line[0], line[1]
            filepath = line[3:]

            if x in ('A', 'M', 'R', 'D') and y == ' ':
                staged.append({'status': x, 'file': filepath})
            elif y in ('M', 'D'):
                modified.append({'status': y, 'file': filepath})
            elif x == '?' and y == '?':
                untracked.append(filepath)

        # Branch info
        _, branch = _run_git(['branch', '--show-current'], self.repo_path)

        return {
            'branch': branch,
            'staged': staged,
            'modified': modified,
            'untracked': untracked,
            'clean': not staged and not modified and not untracked,
        }

    def diff(self, staged: bool = False, file_path: str = '') -> str:
        """Get diff output."""
        args = ['diff']
        if staged:
            args.append('--staged')
        if file_path:
            args.extend(['--', file_path])
        ok, output = _run_git(args, self.repo_path)
        return output if ok else ''

    def diff_stat(self, staged: bool = False) -> Dict[str, Any]:
        """Get diff statistics."""
        args = ['diff', '--stat']
        if staged:
            args.append('--staged')
        ok, output = _run_git(args, self.repo_path)
        if not ok:
            return {'error': output}

        files_changed = 0
        insertions = 0
        deletions = 0

        # Parse the summary line
        for line in output.split('\n'):
            m = re.search(r'(\d+) files? changed', line)
            if m:
                files_changed = int(m.group(1))
            m = re.search(r'(\d+) insertions?\(\+\)', line)
            if m:
                insertions = int(m.group(1))
            m = re.search(r'(\d+) deletions?\(-\)', line)
            if m:
                deletions = int(m.group(1))

        return {
            'files_changed': files_changed,
            'insertions': insertions,
            'deletions': deletions,
        }

    def log(self, n: int = 10) -> List[Dict[str, str]]:
        """Get recent commits."""
        fmt = '--format=%H|%h|%an|%ae|%ar|%s'
        ok, output = _run_git(['log', f'-{n}', fmt], self.repo_path)
        if not ok:
            return []

        commits = []
        for line in output.split('\n'):
            if not line:
                continue
            parts = line.split('|', 5)
            if len(parts) >= 6:
                commits.append({
                    'hash': parts[0],
                    'short_hash': parts[1],
                    'author': parts[2],
                    'email': parts[3],
                    'relative_date': parts[4],
                    'message': parts[5],
                })

        return commits

    def blame(self, file_path: str, line_start: int = 0,
              line_end: int = 0) -> List[Dict[str, str]]:
        """Get blame info for a file."""
        args = ['blame', '--porcelain']
        if line_start > 0 and line_end > 0:
            args.extend([f'-L{line_start},{line_end}'])
        args.append(file_path)

        ok, output = _run_git(args, self.repo_path)
        if not ok:
            return []

        blames = []
        current: Dict[str, str] = {}

        for line in output.split('\n'):
            if line.startswith('\t'):
                current['line'] = line[1:]
                blames.append(current)
                current = {}
            elif ' ' in line:
                key, _, value = line.partition(' ')
                if len(key) == 40:  # SHA
                    current['hash'] = key
                elif key == 'author':
                    current['author'] = value
                elif key == 'summary':
                    current['summary'] = value

        return blames

    def add(self, files: List[str]) -> bool:
        """Stage files for commit."""
        ok, _ = _run_git(['add'] + files, self.repo_path)
        return ok

    def commit(self, message: str) -> Tuple[bool, str]:
        """Create a commit."""
        ok, output = _run_git(['commit', '-m', message], self.repo_path)
        return ok, output

    def current_branch(self) -> str:
        """Get current branch name."""
        ok, output = _run_git(['branch', '--show-current'], self.repo_path)
        return output if ok else ''

    def branches(self) -> List[str]:
        """List all local branches."""
        ok, output = _run_git(['branch', '--list'], self.repo_path)
        if not ok:
            return []
        return [b.strip().lstrip('* ') for b in output.split('\n') if b.strip()]

    def stash(self) -> bool:
        """Stash current changes."""
        ok, _ = _run_git(['stash'], self.repo_path)
        return ok

    def stash_pop(self) -> bool:
        """Pop stashed changes."""
        ok, _ = _run_git(['stash', 'pop'], self.repo_path)
        return ok

    def show_file(self, file_path: str, ref: str = 'HEAD') -> str:
        """Show file contents at a specific ref."""
        ok, output = _run_git(['show', f'{ref}:{file_path}'], self.repo_path)
        return output if ok else ''

    def tags(self) -> List[str]:
        """List all tags."""
        ok, output = _run_git(['tag', '--list'], self.repo_path)
        if not ok:
            return []
        return [t.strip() for t in output.split('\n') if t.strip()]

    def remote_url(self) -> str:
        """Get origin remote URL."""
        ok, output = _run_git(['remote', 'get-url', 'origin'], self.repo_path)
        return output if ok else ''

    def summary(self) -> Dict[str, Any]:
        """Get a full repository summary."""
        status = self.status()
        recent = self.log(5)

        return {
            'branch': status.get('branch', '?'),
            'clean': status.get('clean', False),
            'staged': len(status.get('staged', [])),
            'modified': len(status.get('modified', [])),
            'untracked': len(status.get('untracked', [])),
            'recent_commits': [
                f"{c['short_hash']} {c['message']}" for c in recent
            ],
            'remote': self.remote_url(),
        }
