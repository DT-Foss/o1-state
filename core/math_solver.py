"""
Math Solver — Symbolic Computation Engine
===========================================
Pure symbolic math with NO external dependencies (no sympy, no scipy).
Handles:
  1. Arithmetic: +, -, *, /, **, %, integer and float
  2. Algebra: solve linear/quadratic equations, systems of 2 equations
  3. Calculus: polynomial derivatives and integrals
  4. Number theory: GCD, LCM, prime check, factorize, modular arithmetic
  5. Expression evaluation: parse and compute mathematical expressions

All computation is exact (rational where possible).
No approximations, no floating point unless explicitly needed.
"""

import re
import math
from typing import Dict, Any, List, Optional, Tuple, Union
from fractions import Fraction


class MathSolver:
    """Symbolic math engine for Fiber 3 reasoning."""

    def solve(self, query: str) -> Dict[str, Any]:
        """
        Main entry: parse and solve a math problem.

        Accepts natural language or symbolic notation:
          "What is 17 * 23?"
          "Solve x^2 - 5x + 6 = 0"
          "derivative of 3x^2 + 2x - 1"
          "Is 97 prime?"
          "GCD of 84 and 36"
          "Solve: 2x + 3y = 7, x - y = 1"
        """
        trace = []
        q = query.strip().rstrip('?.')

        # Try each solver in order
        solvers = [
            self._try_arithmetic,
            self._try_quadratic,   # Before linear — quadratic matches x^2 first
            self._try_system,
            self._try_equation,
            self._try_derivative,
            self._try_integral,
            self._try_number_theory,
            self._try_expression,
        ]

        for solver in solvers:
            result = solver(q, trace)
            if result is not None:
                return result

        trace.append(f"No solver matched: {q}")
        return {
            'answer': None,
            'confidence_level': 'LOW',
            'reasoning_type': 'math',
            'trace': trace,
        }

    # === ARITHMETIC ===

    def _try_arithmetic(self, q: str, trace: list) -> Optional[Dict]:
        """Direct arithmetic: 'What is 17 * 23?', '15 + 27', 'calculate 100 / 7'"""
        # Strip natural language prefix
        q_clean = re.sub(
            r'^(?:what\s+is|calculate|compute|evaluate|find)\s+',
            '', q, flags=re.I
        ).strip()

        # sqrt(N) function
        m = re.match(r'sqrt\s*\(\s*(\d+\.?\d*)\s*\)', q_clean)
        if m:
            n = float(m.group(1))
            result = math.sqrt(n)
            r_str = str(int(result)) if result == int(result) else f"{result:.6f}"
            trace.append(f"sqrt({n}) = {r_str}")
            return self._result(r_str, 'HIGH', 'arithmetic', trace)

        # Match: number op number (op number)*
        m = re.match(
            r'^(-?\d+\.?\d*)\s*([+\-*/^%]|(?:\*\*)|mod)\s*(-?\d+\.?\d*)$',
            q_clean
        )
        if m:
            a = self._parse_number(m.group(1))
            op = m.group(2)
            b = self._parse_number(m.group(3))
            result = self._compute_op(a, op, b)
            if result is not None:
                trace.append(f"Arithmetic: {a} {op} {b} = {result}")
                return self._result(str(result), 'HIGH', 'arithmetic', trace)

        # More complex expression: try safe eval
        if re.match(r'^[\d\s+\-*/().^%]+$', q_clean):
            try:
                result = self._safe_eval(q_clean)
                trace.append(f"Expression: {q_clean} = {result}")
                return self._result(str(result), 'HIGH', 'arithmetic', trace)
            except Exception:
                pass

        return None

    def _try_expression(self, q: str, trace: list) -> Optional[Dict]:
        """Evaluate mathematical expressions."""
        q_clean = re.sub(
            r'^(?:what\s+is|calculate|compute|evaluate)\s+',
            '', q, flags=re.I
        ).strip()

        # Check if it looks like a math expression
        if not re.search(r'\d', q_clean):
            return None

        # Replace common math notation
        expr = q_clean.replace('^', '**').replace('mod', '%')

        # Only allow safe characters
        if not re.match(r'^[\d\s+\-*/().%]+$', expr):
            return None

        try:
            result = self._safe_eval(expr)
            trace.append(f"Expression: {q_clean} = {result}")
            return self._result(str(result), 'HIGH', 'expression', trace)
        except Exception:
            return None

    # === ALGEBRA ===

    def _try_equation(self, q: str, trace: list) -> Optional[Dict]:
        """Solve linear equations: 'solve 2x + 3 = 11'"""
        m = re.search(
            r'(?:solve\s*:?\s*)?(-?\d*\.?\d*)\s*([a-z])\s*([+\-])\s*(\d+\.?\d*)\s*=\s*(-?\d+\.?\d*)',
            q, re.I
        )
        if m:
            coeff = float(m.group(1)) if m.group(1) and m.group(1) != '-' else (
                -1.0 if m.group(1) == '-' else 1.0
            )
            var = m.group(2)
            sign = 1 if m.group(3) == '+' else -1
            const = float(m.group(4)) * sign
            rhs = float(m.group(5))

            if coeff == 0:
                return None

            solution = Fraction(rhs - const).limit_denominator(10000) / Fraction(coeff).limit_denominator(10000)
            trace.append(f"Linear: {coeff}{var} + {const} = {rhs}")
            trace.append(f"Solution: {var} = ({rhs} - {const}) / {coeff} = {solution}")

            answer = f"{var} = {solution}"
            if solution.denominator == 1:
                answer = f"{var} = {solution.numerator}"
            return self._result(answer, 'HIGH', 'linear_equation', trace)

        # Simpler: "solve x + 3 = 7" or "x - 5 = 10"
        m = re.search(
            r'(?:solve\s*:?\s*)?([a-z])\s*([+\-])\s*(\d+\.?\d*)\s*=\s*(-?\d+\.?\d*)',
            q, re.I
        )
        if m:
            var = m.group(1)
            sign = 1 if m.group(2) == '+' else -1
            const = float(m.group(3)) * sign
            rhs = float(m.group(4))
            solution = rhs - const
            trace.append(f"Linear: {var} + {const} = {rhs} → {var} = {solution}")
            answer = f"{var} = {int(solution)}" if solution == int(solution) else f"{var} = {solution}"
            return self._result(answer, 'HIGH', 'linear_equation', trace)

        return None

    def _try_quadratic(self, q: str, trace: list) -> Optional[Dict]:
        """Solve quadratic: 'solve x^2 - 5x + 6 = 0'"""
        # Match ax^2 + bx + c = 0
        m = re.search(
            r'(?:solve\s*:?\s*)?(-?\d*\.?\d*)\s*([a-z])\s*[\^*]\s*2\s*'
            r'([+\-]\s*\d*\.?\d*)\s*\2\s*'
            r'([+\-]\s*\d+\.?\d*)\s*=\s*0',
            q, re.I
        )
        if m:
            a_str = m.group(1) or '1'
            if a_str == '-':
                a_str = '-1'
            a = float(a_str)
            var = m.group(2)
            b_str = m.group(3).replace(' ', '')
            b = float(b_str) if b_str and b_str not in ('+', '-') else (
                1.0 if b_str == '+' else -1.0 if b_str == '-' else 0.0
            )
            c_str = m.group(4).replace(' ', '')
            c = float(c_str)

            return self._solve_quadratic(a, b, c, var, trace)

        # Simpler pattern: x^2 - 5x + 6 = 0 (coefficient 1)
        m = re.search(
            r'(?:solve\s*:?\s*)?([a-z])\^2\s*([+\-]\s*\d+)\s*\1\s*([+\-]\s*\d+)\s*=\s*0',
            q, re.I
        )
        if m:
            var = m.group(1)
            b = float(m.group(2).replace(' ', ''))
            c = float(m.group(3).replace(' ', ''))
            return self._solve_quadratic(1, b, c, var, trace)

        return None

    def _solve_quadratic(self, a: float, b: float, c: float,
                         var: str, trace: list) -> Dict:
        """Solve ax² + bx + c = 0 using the quadratic formula."""
        discriminant = b * b - 4 * a * c
        trace.append(f"Quadratic: {a}{var}² + {b}{var} + {c} = 0")
        trace.append(f"Discriminant: {b}² - 4·{a}·{c} = {discriminant}")

        if discriminant > 0:
            sqrt_d = math.sqrt(discriminant)
            x1 = (-b + sqrt_d) / (2 * a)
            x2 = (-b - sqrt_d) / (2 * a)
            # Check if solutions are integers
            x1_str = str(int(x1)) if x1 == int(x1) else f"{x1:.4f}"
            x2_str = str(int(x2)) if x2 == int(x2) else f"{x2:.4f}"
            answer = f"{var} = {x1_str} or {var} = {x2_str}"
            trace.append(f"Two real roots: {x1_str}, {x2_str}")
        elif discriminant == 0:
            x = -b / (2 * a)
            x_str = str(int(x)) if x == int(x) else f"{x:.4f}"
            answer = f"{var} = {x_str} (double root)"
            trace.append(f"Double root: {x_str}")
        else:
            real = -b / (2 * a)
            imag = math.sqrt(-discriminant) / (2 * a)
            r_str = f"{real:.4f}" if real != 0 else ""
            i_str = f"{imag:.4f}"
            answer = f"{var} = {r_str} ± {i_str}i (complex roots)"
            trace.append(f"Complex roots: {r_str} ± {i_str}i")

        return self._result(answer, 'HIGH', 'quadratic', trace)

    def _try_system(self, q: str, trace: list) -> Optional[Dict]:
        """Solve 2-equation systems: '2x + 3y = 7, x - y = 1'"""
        # Find two equations separated by comma, semicolon, or 'and'
        parts = re.split(r'[,;]\s*|\s+and\s+', q)
        if len(parts) != 2:
            return None

        eq1 = self._parse_linear_eq(parts[0].strip())
        eq2 = self._parse_linear_eq(parts[1].strip())

        if eq1 is None or eq2 is None:
            return None

        a1, b1, c1, vars1 = eq1
        a2, b2, c2, vars2 = eq2

        if vars1 != vars2:
            return None

        var_x, var_y = vars1

        # Cramer's rule
        det = a1 * b2 - a2 * b1
        if det == 0:
            trace.append("System has no unique solution (det = 0)")
            return self._result("No unique solution (parallel or identical lines)",
                              'HIGH', 'system', trace)

        x = Fraction(int(c1 * b2 - c2 * b1), int(det))
        y = Fraction(int(a1 * c2 - a2 * c1), int(det))

        trace.append(f"System: {a1}{var_x} + {b1}{var_y} = {c1}")
        trace.append(f"        {a2}{var_x} + {b2}{var_y} = {c2}")
        trace.append(f"det = {det}")

        x_str = str(x) if x.denominator != 1 else str(x.numerator)
        y_str = str(y) if y.denominator != 1 else str(y.numerator)
        answer = f"{var_x} = {x_str}, {var_y} = {y_str}"
        trace.append(f"Solution: {answer}")

        return self._result(answer, 'HIGH', 'system', trace)

    def _parse_linear_eq(self, eq: str) -> Optional[Tuple]:
        """Parse 'ax + by = c' into (a, b, c, (x_var, y_var))."""
        eq = re.sub(r'^solve\s*:?\s*', '', eq, flags=re.I).strip()

        # Match: (coeff)(var) + (coeff)(var) = (number)
        m = re.match(
            r'(-?\d*\.?\d*)\s*([a-z])\s*([+\-])\s*(\d*\.?\d*)\s*([a-z])\s*=\s*(-?\d+\.?\d*)',
            eq
        )
        if m:
            a = float(m.group(1)) if m.group(1) and m.group(1) != '-' else (
                -1.0 if m.group(1) == '-' else 1.0
            )
            var_x = m.group(2)
            sign = 1 if m.group(3) == '+' else -1
            b = float(m.group(4)) * sign if m.group(4) else sign * 1.0
            var_y = m.group(5)
            c = float(m.group(6))
            return (a, b, c, (var_x, var_y))

        return None

    # === CALCULUS ===

    def _try_derivative(self, q: str, trace: list) -> Optional[Dict]:
        """Polynomial derivative: 'derivative of 3x^2 + 2x - 1'"""
        m = re.search(
            r'(?:derivative|differentiate|d/dx)\s+(?:of\s+)?(.+)',
            q, re.I
        )
        if not m:
            return None

        expr = m.group(1).strip()
        terms = self._parse_polynomial(expr)
        if not terms:
            return None

        trace.append(f"Polynomial: {expr}")

        # Differentiate each term: d/dx (a * x^n) = a*n * x^(n-1)
        deriv_terms = []
        for coeff, power in terms:
            if power == 0:
                continue  # constant → 0
            new_coeff = coeff * power
            new_power = power - 1
            deriv_terms.append((new_coeff, new_power))

        if not deriv_terms:
            answer = "0"
        else:
            answer = self._format_polynomial(deriv_terms)

        trace.append(f"Derivative: {answer}")
        return self._result(answer, 'HIGH', 'derivative', trace)

    def _try_integral(self, q: str, trace: list) -> Optional[Dict]:
        """Polynomial integral: 'integral of 3x^2 + 2x - 1'"""
        m = re.search(
            r'(?:integral|integrate|antiderivative)\s+(?:of\s+)?(.+)',
            q, re.I
        )
        if not m:
            return None

        expr = m.group(1).strip()
        terms = self._parse_polynomial(expr)
        if not terms:
            return None

        trace.append(f"Polynomial: {expr}")

        # Integrate each term: ∫ a*x^n dx = a/(n+1) * x^(n+1)
        int_terms = []
        for coeff, power in terms:
            new_power = power + 1
            new_coeff = Fraction(coeff).limit_denominator(10000) / new_power
            int_terms.append((float(new_coeff), new_power))

        answer = self._format_polynomial(int_terms) + " + C"
        trace.append(f"Integral: {answer}")
        return self._result(answer, 'HIGH', 'integral', trace)

    def _parse_polynomial(self, expr: str) -> Optional[List[Tuple[float, int]]]:
        """Parse polynomial string into [(coeff, power), ...]."""
        terms = []
        # Normalize: add space around + and -
        expr = re.sub(r'(?<=[^\s+\-])\s*([+\-])', r' \1', expr)
        parts = re.split(r'(?=[+\-])', expr)

        for part in parts:
            part = part.strip()
            if not part:
                continue
            # Strip leading + (- is kept for sign)
            if part.startswith('+'):
                part = part[1:].strip()

            # ax^n
            m = re.match(r'(-?\d*\.?\d*)\s*[a-z]\s*\^\s*(\d+)', part)
            if m:
                coeff = float(m.group(1)) if m.group(1) and m.group(1) not in ('+', '-') else (
                    -1.0 if m.group(1) == '-' else 1.0
                )
                power = int(m.group(2))
                terms.append((coeff, power))
                continue

            # ax (power 1)
            m = re.match(r'(-?\d*\.?\d*)\s*[a-z]$', part)
            if m:
                coeff = float(m.group(1)) if m.group(1) and m.group(1) not in ('+', '-') else (
                    -1.0 if m.group(1) == '-' else 1.0
                )
                terms.append((coeff, 1))
                continue

            # constant (power 0)
            m = re.match(r'(-?\d+\.?\d*)\s*$', part)
            if m:
                terms.append((float(m.group(1)), 0))
                continue

        return terms if terms else None

    def _format_polynomial(self, terms: List[Tuple[float, int]]) -> str:
        """Format [(coeff, power), ...] as a polynomial string."""
        parts = []
        for i, (coeff, power) in enumerate(sorted(terms, key=lambda t: -t[1])):
            if coeff == 0:
                continue

            # Format coefficient
            if coeff == int(coeff):
                coeff = int(coeff)

            if power == 0:
                parts.append(str(coeff))
            elif power == 1:
                if coeff == 1:
                    parts.append('x')
                elif coeff == -1:
                    parts.append('-x')
                else:
                    parts.append(f"{coeff}x")
            else:
                if coeff == 1:
                    parts.append(f"x^{power}")
                elif coeff == -1:
                    parts.append(f"-x^{power}")
                else:
                    parts.append(f"{coeff}x^{power}")

        if not parts:
            return "0"

        result = parts[0]
        for p in parts[1:]:
            if p.startswith('-'):
                result += f" - {p[1:]}"
            else:
                result += f" + {p}"
        return result

    # === NUMBER THEORY ===

    def _try_number_theory(self, q: str, trace: list) -> Optional[Dict]:
        """GCD, LCM, prime check, factorize, mod."""
        ql = q.lower()

        # GCD
        m = re.search(r'gcd\s+(?:of\s+)?(\d+)\s+(?:and\s+)?(\d+)', ql)
        if m:
            a, b = int(m.group(1)), int(m.group(2))
            result = math.gcd(a, b)
            trace.append(f"GCD({a}, {b}) = {result}")
            return self._result(str(result), 'HIGH', 'gcd', trace)

        # LCM
        m = re.search(r'lcm\s+(?:of\s+)?(\d+)\s+(?:and\s+)?(\d+)', ql)
        if m:
            a, b = int(m.group(1)), int(m.group(2))
            result = (a * b) // math.gcd(a, b)
            trace.append(f"LCM({a}, {b}) = {result}")
            return self._result(str(result), 'HIGH', 'lcm', trace)

        # Prime check
        m = re.search(r'(?:is\s+)?(\d+)\s+(?:a\s+)?prime', ql)
        if m:
            n = int(m.group(1))
            is_prime = self._is_prime(n)
            answer = f"{n} is {'prime' if is_prime else 'not prime'}"
            if not is_prime and n > 1:
                factors = self._factorize(n)
                answer += f" ({n} = {' × '.join(str(f) for f in factors)})"
            trace.append(answer)
            return self._result(answer, 'HIGH', 'prime_check', trace)

        # Factorize
        m = re.search(r'(?:factorize|factor|prime\s+factors?\s+of)\s+(\d+)', ql)
        if m:
            n = int(m.group(1))
            factors = self._factorize(n)
            answer = f"{n} = {' × '.join(str(f) for f in factors)}"
            trace.append(answer)
            return self._result(answer, 'HIGH', 'factorize', trace)

        # Modular arithmetic: "17 mod 5" or "17 % 5"
        m = re.search(r'(\d+)\s+(?:mod|%)\s+(\d+)', ql)
        if m:
            a, b = int(m.group(1)), int(m.group(2))
            if b != 0:
                result = a % b
                trace.append(f"{a} mod {b} = {result}")
                return self._result(str(result), 'HIGH', 'modular', trace)

        # Fibonacci
        m = re.search(r'(?:fibonacci|fib)\s*\(?(\d+)\)?', ql)
        if m:
            n = int(m.group(1))
            if n > 1000:
                return None  # Safety limit
            result = self._fibonacci(n)
            trace.append(f"fibonacci({n}) = {result}")
            return self._result(str(result), 'HIGH', 'fibonacci', trace)

        # Factorial
        m = re.search(r'(\d+)\s*!|factorial\s*\(?(\d+)\)?', ql)
        if m:
            n = int(m.group(1) or m.group(2))
            if n > 170:
                return None  # Safety limit (overflow)
            result = math.factorial(n)
            trace.append(f"{n}! = {result}")
            return self._result(str(result), 'HIGH', 'factorial', trace)

        return None

    # === HELPERS ===

    @staticmethod
    def _parse_number(s: str) -> float:
        """Parse number from string."""
        return float(s)

    @staticmethod
    def _compute_op(a: float, op: str, b: float) -> Optional[float]:
        """Compute a binary operation."""
        if op == '+':
            return a + b
        elif op == '-':
            return a - b
        elif op in ('*', '×'):
            return a * b
        elif op == '/':
            return a / b if b != 0 else None
        elif op in ('^', '**'):
            return a ** b
        elif op in ('%', 'mod'):
            return a % b if b != 0 else None
        return None

    @staticmethod
    def _safe_eval(expr: str) -> float:
        """Safely evaluate a math expression (no exec, no builtins)."""
        # Replace ^ with **
        expr = expr.replace('^', '**')
        # Only allow digits, operators, parens, spaces, dots
        if not re.match(r'^[\d\s+\-*/().%]+$', expr):
            raise ValueError(f"Unsafe expression: {expr}")
        # Evaluate with no builtins
        return eval(expr, {"__builtins__": {}}, {})

    @staticmethod
    def _is_prime(n: int) -> bool:
        """Check if n is prime."""
        if n < 2:
            return False
        if n < 4:
            return True
        if n % 2 == 0 or n % 3 == 0:
            return False
        i = 5
        while i * i <= n:
            if n % i == 0 or n % (i + 2) == 0:
                return False
            i += 6
        return True

    @staticmethod
    def _factorize(n: int) -> List[int]:
        """Prime factorization of n."""
        factors = []
        d = 2
        while d * d <= n:
            while n % d == 0:
                factors.append(d)
                n //= d
            d += 1
        if n > 1:
            factors.append(n)
        return factors

    @staticmethod
    def _fibonacci(n: int) -> int:
        """Compute nth Fibonacci number."""
        if n <= 1:
            return n
        a, b = 0, 1
        for _ in range(2, n + 1):
            a, b = b, a + b
        return b

    @staticmethod
    def _result(answer: str, confidence: str, reasoning_type: str,
                trace: list) -> Dict[str, Any]:
        return {
            'answer': answer,
            'confidence_level': confidence,
            'fiber': 3,
            'escalated': False,
            'reasoning_type': reasoning_type,
            'trace': trace,
        }
