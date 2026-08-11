"""
FORMEL-DYNAMIKEN, auf den Stream übertragen -- die zwei aus dem
FERTIG-codex-lab-Rennen, die im Organismus-Regime (ein Leben, ein Pass,
Gate, getragener Zustand) überhaupt eine Entsprechung haben:

  RapidityAdam -- Möbius-Momentum. Re-implementiert nach FERTIG/
      _codex_lab/training_dynamics/race_dynamics.py (Foss: MarkovChains ->
      Minkowski; f(lambda,v) = (lambda+v)/(1+lambda*v)): der Impuls
      akkumuliert in der RAPIDITÄT w (additiv, unbeschränkt), der
      tatsächliche Schritt ist v = tanh(w) -- Lorentz-Geschwindigkeits-
      grenze |v| < 1 mal lr. Bei kleinem w ist tanh(w) ~ w (Standard-
      Momentum auf dem Adam-normalisierten Gradienten); großes w saturiert
      statt zu overshooten. Dieselbe Möbius-Kopplung, aus der der GSSM-Kern
      des Repos gebaut ist, jetzt auf der OPTIMIERER-Seite.

  lorentz_lr -- tau-Lorentz-Schedule (Foss: Collapse Is Contraction):
      lr(t) = lr0 * (gamma(tau(t)) - 1), gamma(tau) = (1 - tau^2)^(-1/2),
      tau linear tau0 -> tau1 ("Kühlung"). Auf das Organismus-Leben
      abgebildet über den Chunk-Index; lr0 so gewählt, dass der PEAK der
      Kurve die Baseline-LR des body-Laufs trifft (3e-4) -- Vergleich bei
      gleicher Spitzenlast, nicht bei gleichem Integral (das Integral IST
      die These des Schedules).

Nicht übertragen, mit Grund (ehrliche Ablehnung statt stiller Lücke):
BvN-Random-Reshuffling braucht Epochen -- ein Leben hat keinen zweiten
Pass. PS-Lifted-Gradient-Consensus braucht viele Micro-Gradienten -- der
Stream-Chunk hat B=1; bei K<=4 Splits ist Konsens-nach-R-Runden vom
exakten Mittel nicht unterscheidbar (Theater, kein Test). FLCA-Router
setzt Letzteres voraus.
"""

import torch


class RapidityAdam(torch.optim.Optimizer):
    """Adam-Statistiken (m, v wie üblich, bias-korrigiert), aber der
    Schritt läuft durch die Rapiditäts-Akkumulation:

        w <- beta1 * w + m_hat / (sqrt(v_hat) + eps)
        p <- p - lr * tanh(w)

    Re-Implementierung (nicht Kopie) der race_dynamics.py-Klasse gleichen
    Namens; Formeln identisch, hier mit eigenem Test (test_dynamics.py:
    Schrittnorm strikt < lr, Rapidität wächst additiv)."""

    def __init__(self, params, lr=3e-4, betas=(0.9, 0.999), eps=1e-8):
        super().__init__(params, dict(lr=lr, betas=betas, eps=eps))

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                st = self.state[p]
                if not st:
                    st["step"] = 0
                    st["w"] = torch.zeros_like(p)
                    st["m"] = torch.zeros_like(p)
                    st["v"] = torch.zeros_like(p)
                st["step"] += 1
                t = st["step"]
                st["m"].mul_(beta1).add_(g, alpha=1 - beta1)
                st["v"].mul_(beta2).addcmul_(g, g, value=1 - beta2)
                m_hat = st["m"] / (1 - beta1 ** t)
                v_hat = st["v"] / (1 - beta2 ** t)
                st["w"].mul_(beta1).add_(m_hat / (v_hat.sqrt().add_(eps)))
                p.add_(torch.tanh(st["w"]), alpha=-group["lr"])
        return loss


def lorentz_lr(step: int, total: int, peak_lr: float = 3e-4,
               tau0: float = 0.95, tau1: float = 0.5) -> float:
    """lr(t) = lr0*(gamma(tau)-1), tau linear tau0 -> tau1. lr0 wird aus
    peak_lr abgeleitet: das Maximum der Kurve liegt bei tau0 (gamma dort
    am größten), also lr0 = peak_lr / (gamma(tau0)-1) -- der Schedule
    startet auf Baseline-Spitze und kühlt auf ~7% davon ab."""
    tau = tau0 - (tau0 - tau1) * (min(step, total) / max(total, 1))
    gamma0 = (1.0 - tau0 ** 2) ** -0.5
    lr0 = peak_lr / (gamma0 - 1.0)
    return lr0 * ((1.0 - tau ** 2) ** -0.5 - 1.0)
