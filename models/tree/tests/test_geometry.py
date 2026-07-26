"""Geometry primitives (Eqs E1/E2/B2) — angle stability, gap sign, containment."""
import math
import torch

from models.tree.geometry import (
    project_to_sphere, stable_angle, angle_arccos, cone_contains,
    angular_margin_ratio, cone_gap, theta_from_logit,
)


def test_project_to_sphere_unit_norm():
    x = torch.randn(10, 64)
    z = project_to_sphere(x)
    assert torch.allclose(z.norm(dim=-1), torch.ones(10), atol=1e-5)


def test_project_to_sphere_zero_safe():
    z = project_to_sphere(torch.zeros(1, 8))
    assert torch.isfinite(z).all()


def test_stable_angle_known_values():
    a = torch.tensor([[1.0, 0.0, 0.0]])
    b = torch.tensor([[1.0, 0.0, 0.0]])
    c = torch.tensor([[0.0, 1.0, 0.0]])
    d = torch.tensor([[-1.0, 0.0, 0.0]])
    assert float(stable_angle(a, b)) < 1e-5                      # 0
    assert abs(float(stable_angle(a, c)) - math.pi / 2) < 1e-4   # 90 deg
    assert abs(float(stable_angle(a, d)) - math.pi) < 1e-3       # 180 deg


def test_stable_angle_matches_arccos_midrange():
    a = torch.randn(20, 16)
    b = torch.randn(20, 16)
    s = stable_angle(a, b)
    r = angle_arccos(a, b)
    assert torch.allclose(s, r, atol=1e-4)


def test_stable_angle_grad_finite_at_poles():
    # arccos grad diverges at +/-1; the atan2 form must stay finite.
    a = torch.tensor([[1.0, 0.0, 0.0]], requires_grad=True)
    b = torch.tensor([[1.0, 0.0, 0.0]])          # coincident -> pole
    ang = stable_angle(a, b)
    ang.backward()
    assert torch.isfinite(a.grad).all()

    a2 = torch.tensor([[1.0, 0.0, 0.0]], requires_grad=True)
    b2 = torch.tensor([[-1.0, 0.0, 0.0]])        # antipodal -> other pole
    stable_angle(a2, b2).backward()
    assert torch.isfinite(a2.grad).all()


def test_theta_from_logit_range():
    a = torch.linspace(-10, 10, 50)
    th = theta_from_logit(a, math.pi / 2)
    assert (th > 0).all() and (th < math.pi / 2).all()
    assert abs(float(theta_from_logit(torch.zeros(()), math.pi / 2)) - math.pi / 4) < 1e-6


def test_cone_contains():
    mu = torch.tensor([1.0, 0.0, 0.0])
    z_in = project_to_sphere(torch.tensor([1.0, 0.1, 0.0]))
    z_out = torch.tensor([0.0, 1.0, 0.0])
    theta = torch.tensor(0.3)
    assert bool(cone_contains(z_in, mu, theta))
    assert not bool(cone_contains(z_out, mu, theta))


def test_angular_margin_ratio_gate():
    mu = torch.tensor([1.0, 0.0, 0.0])
    theta = torch.tensor(math.pi / 4)
    z_in = project_to_sphere(torch.tensor([1.0, 0.2, 0.0]))
    z_out = torch.tensor([0.0, 1.0, 0.0])
    assert float(angular_margin_ratio(z_in, mu, theta)) <= 1.0
    assert float(angular_margin_ratio(z_out, mu, theta)) > 1.0


def test_cone_gap_sign_and_noinf():
    # parent wide, one narrow child close to axis -> positive annulus
    mu_n = torch.tensor([1.0, 0.0, 0.0])
    theta_n = torch.tensor(1.2)
    mu_c = project_to_sphere(torch.tensor([[1.0, 0.05, 0.0]]))
    theta_c = torch.tensor([0.2])
    gap = cone_gap(theta_n, mu_n, mu_c, theta_c)
    assert float(gap) > 0

    # child almost fills parent -> gap can go negative (overlap violation)
    theta_c_big = torch.tensor([1.15])
    gap2 = cone_gap(theta_n, mu_n, mu_c, theta_c_big)
    assert float(gap2) < float(gap)


def test_cone_gap_no_children_is_inf():
    g = cone_gap(torch.tensor(1.0), torch.tensor([1.0, 0.0]),
                 torch.zeros(0, 2), torch.zeros(0))
    assert math.isinf(float(g))
