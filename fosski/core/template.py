"""
Template Engine — Mustache-Style Text Generation
==================================================
Variable substitution, conditionals, loops, partials.
For generating code, reports, emails, and structured output.

Syntax:
  {{variable}}           — simple substitution
  {{#section}}...{{/section}}  — conditional block (truthy)
  {{^section}}...{{/section}}  — inverted block (falsy)
  {{#items}}...{{/items}}      — loop over list
  {{>partial}}           — include partial template
  {{{raw}}}              — unescaped output

This replaces Jinja2/Mako without the dependency.
"""

import re
from typing import Dict, Any, List, Optional, Callable


class TemplateEngine:
    """
    Mustache-compatible template engine.

    Usage:
        engine = TemplateEngine()
        result = engine.render("Hello, {{name}}!", {'name': 'World'})
        # → "Hello, World!"

        result = engine.render(
            "{{#items}}  - {{name}}: {{value}}\\n{{/items}}",
            {'items': [{'name': 'a', 'value': 1}, {'name': 'b', 'value': 2}]}
        )
    """

    def __init__(self):
        self.partials: Dict[str, str] = {}
        self.helpers: Dict[str, Callable] = {}
        self._register_default_helpers()

    def render(self, template: str, context: Dict[str, Any],
               partials: Optional[Dict[str, str]] = None) -> str:
        """
        Render a template with the given context.

        Args:
            template: template string
            context: variable bindings
            partials: named partial templates

        Returns:
            rendered string
        """
        if partials:
            merged_partials = {**self.partials, **partials}
        else:
            merged_partials = self.partials

        return self._render(template, context, merged_partials)

    def _render(self, template: str, context: Dict[str, Any],
                partials: Dict[str, str]) -> str:
        """Internal recursive render."""
        result = template

        # Step 1: Partials {{>name}}
        result = re.sub(
            r'\{\{>\s*(\w+)\s*\}\}',
            lambda m: self._render(
                partials.get(m.group(1), ''),
                context, partials
            ),
            result
        )

        # Step 2: Sections {{#section}}...{{/section}}
        result = self._process_sections(result, context, partials)

        # Step 3: Inverted sections {{^section}}...{{/section}}
        result = self._process_inverted_sections(result, context)

        # Step 4: Helpers {{helper arg1 arg2}}
        result = self._process_helpers(result, context)

        # Step 5: Raw variables {{{var}}}
        result = re.sub(
            r'\{\{\{\s*([\w.]+)\s*\}\}\}',
            lambda m: str(self._lookup(m.group(1), context)),
            result
        )

        # Step 6: Escaped variables {{var}}
        result = re.sub(
            r'\{\{\s*([\w.]+)\s*\}\}',
            lambda m: _html_escape(str(self._lookup(m.group(1), context))),
            result
        )

        return result

    def _process_sections(self, template: str, context: Dict[str, Any],
                          partials: Dict[str, str]) -> str:
        """Process {{#section}}...{{/section}} blocks."""
        pattern = r'\{\{#\s*(\w+)\s*\}\}(.*?)\{\{/\s*\1\s*\}\}'

        def replace_section(m):
            key = m.group(1)
            body = m.group(2)
            value = context.get(key)

            if value is None or value is False or value == '' or value == 0:
                return ''

            if isinstance(value, list):
                # Loop
                parts = []
                for item in value:
                    if isinstance(item, dict):
                        merged = {**context, **item}
                        parts.append(self._render(body, merged, partials))
                    else:
                        merged = {**context, '.': item, 'this': item}
                        parts.append(self._render(body, merged, partials))
                return ''.join(parts)

            if isinstance(value, dict):
                merged = {**context, **value}
                return self._render(body, merged, partials)

            # Truthy value: render block once
            return self._render(body, context, partials)

        return re.sub(pattern, replace_section, template, flags=re.S)

    def _process_inverted_sections(self, template: str,
                                   context: Dict[str, Any]) -> str:
        """Process {{^section}}...{{/section}} blocks (render when falsy)."""
        pattern = r'\{\{\^\s*(\w+)\s*\}\}(.*?)\{\{/\s*\1\s*\}\}'

        def replace_inverted(m):
            key = m.group(1)
            body = m.group(2)
            value = context.get(key)

            if value is None or value is False or value == '' or \
               (isinstance(value, list) and len(value) == 0):
                return body
            return ''

        return re.sub(pattern, replace_inverted, template, flags=re.S)

    def _process_helpers(self, template: str, context: Dict[str, Any]) -> str:
        """Process {{helper arg1 arg2}} calls."""
        def replace_helper(m):
            parts = m.group(1).strip().split()
            if not parts:
                return m.group(0)
            name = parts[0]
            if name not in self.helpers:
                return m.group(0)
            args = [self._lookup(a, context) if not a.startswith('"') else a.strip('"')
                    for a in parts[1:]]
            try:
                return str(self.helpers[name](*args))
            except Exception:
                return m.group(0)

        return re.sub(r'\{\{\s*(\w+(?:\s+[^\}]+)?)\s*\}\}', replace_helper, template)

    def _lookup(self, key: str, context: Dict[str, Any]) -> Any:
        """Look up a dotted key in context."""
        parts = key.split('.')
        current = context
        for part in parts:
            if isinstance(current, dict):
                current = current.get(part, '')
            else:
                return ''
        return current if current is not None else ''

    def register_partial(self, name: str, template: str):
        """Register a reusable partial template."""
        self.partials[name] = template

    def register_helper(self, name: str, fn: Callable):
        """Register a helper function."""
        self.helpers[name] = fn

    def _register_default_helpers(self):
        """Built-in helpers."""
        self.helpers['upper'] = lambda s: str(s).upper()
        self.helpers['lower'] = lambda s: str(s).lower()
        self.helpers['capitalize'] = lambda s: str(s).capitalize()
        self.helpers['len'] = lambda s: str(len(s)) if hasattr(s, '__len__') else '0'
        self.helpers['default'] = lambda val, default: str(val) if val else str(default)
        self.helpers['join'] = lambda lst, sep=', ': sep.join(str(x) for x in lst) if isinstance(lst, list) else str(lst)


# === Predefined Templates ===

TEMPLATES = {
    'python_function': '''def {{name}}({{params}}):
    """{{docstring}}"""
{{#body}}    {{.}}
{{/body}}    return {{return_value}}
''',

    'python_class': '''class {{name}}{{#parent}}({{parent}}){{/parent}}:
    """{{docstring}}"""

    def __init__(self{{#init_params}}, {{.}}{{/init_params}}):
{{#init_params}}        self.{{.}} = {{.}}
{{/init_params}}
''',

    'html_page': '''<!DOCTYPE html>
<html lang="{{lang}}">
<head>
    <meta charset="UTF-8">
    <title>{{title}}</title>
{{#styles}}    <link rel="stylesheet" href="{{.}}">
{{/styles}}</head>
<body>
    <h1>{{title}}</h1>
    {{{content}}}
</body>
</html>
''',

    'markdown_report': '''# {{title}}

{{description}}

{{#sections}}
## {{heading}}

{{content}}

{{/sections}}

---
Generated by FOSS-KI
''',

    'email': '''From: {{from}}
To: {{to}}
Subject: {{subject}}

{{greeting}} {{recipient}},

{{body}}

{{closing}},
{{sender}}
''',

    'json_config': '''{
{{#items}}    "{{key}}": {{value}}{{#comma}},{{/comma}}
{{/items}}}
''',

    'csv_row': '{{#fields}}{{.}},{{/fields}}',

    'test_case': '''def test_{{name}}():
    """{{description}}"""
    # Arrange
    {{arrange}}

    # Act
    result = {{act}}

    # Assert
    assert {{assertion}}
''',

    'changelog': '''## {{version}} ({{date}})

{{#changes}}
### {{type}}
{{#items}}- {{.}}
{{/items}}
{{/changes}}
''',

    'dockerfile': '''FROM {{base_image}}

WORKDIR {{workdir}}

{{#copy_files}}COPY {{.}} .
{{/copy_files}}
{{#run_commands}}RUN {{.}}
{{/run_commands}}
{{#env_vars}}ENV {{key}}={{value}}
{{/env_vars}}
{{#expose}}EXPOSE {{.}}
{{/expose}}
CMD [{{cmd}}]
''',
}


def _html_escape(text: str) -> str:
    """Escape HTML entities."""
    return (text
            .replace('&', '&amp;')
            .replace('<', '&lt;')
            .replace('>', '&gt;')
            .replace('"', '&quot;')
            .replace("'", '&#x27;'))


def render_template(name: str, context: Dict[str, Any]) -> str:
    """Render a predefined template by name."""
    template = TEMPLATES.get(name)
    if not template:
        return f"Unknown template: {name}"
    engine = TemplateEngine()
    return engine.render(template, context)
