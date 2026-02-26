import sets
import torch
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
import numpy as np

def sample_zonotope(center, G, n=1000):
    """
    center: (B, d)
    G: (B, d, k)
    """
    B, d, k = G.shape
    xi = torch.rand(B, n, k, device=G.device) * 2 - 1
    pts = center[:, None, :] + torch.einsum("bdk,bnk->bnd", G, xi)
    return pts


def inside_polytope(A, b, x, tol=1e-6):
    """
    A: (B, N, d)
    b: (B, N)
    x: (B, n, d)
    """
    lhs = torch.einsum("bnd,bkd->bnk", x, A)
    return (lhs <= b[:, None, :] + tol).all(dim=-1)


def inside_zonotope(center, G, x):
    """
    Check if x ∈ {c + G [-1,1]^k}
    by solving least squares and checking ||xi||_inf <= 1
    """
    B, n, d = x.shape
    k = G.shape[2]

    inside = torch.ones(B, n, dtype=torch.bool, device=x.device)

    for b in range(B):
        Gb = G[b]
        cb = center[b]
        for i in range(n):
            y = x[b, i] - cb
            xi, *_ = torch.linalg.lstsq(Gb, y)
            inside[b, i] = (xi.abs().max() <= 1.0001)

    return inside

def test_polytope_matches_zonotope(env):
    """
    env must be initialized with:
    - env.polytope = True/False
    - env.safe_action_set()
    """
    torch.manual_seed(0)

    # ---- Compute zonotope version ----
    env.polytope = False
    env.shape = sets.Zonotope
    Z = env.safe_action_set()
    center = Z.center 
    G = Z.generator 

    # ---- Compute polytope version ----
    env.polytope = True
    env.shape = sets.HPolytope
    P = env.safe_action_set()
    A = P.A
    b = P.b

    same = True
    visualize_sets(center, G, A, b, 10000)

    # ---- Sample points from zonotope ----
    pts_Z = sample_zonotope(center, G, n=10000)

    # ---- Check zonotope ⊆ polytope ----
    ok1 = inside_polytope(A, b, pts_Z)
    
    viol = ~ok1
    if viol.any():
        num_viol = viol.sum().item()
        total = viol.numel()

        print(f"❌ {num_viol} / {total} zonotope points are outside the polytope "
            f"({100*num_viol/total:.2f}%)")
        bi = 0  # batch index
        idx = viol[bi].nonzero()[0].item()

        x = pts_Z[bi, idx]

        print("❌ Violating zonotope point:", x)

        lhs = (A[bi] @ x)        # (num_constraints,)
        diff = lhs - b[bi]       # (num_constraints,)

        worst = torch.argmax(diff)

        print("Worst constraint index:", worst.item())
        print("A[i]:", A[bi, worst])
        print("b[i]:", b[bi, worst])
        print("A x:", lhs[worst])
        print("Violation:", diff[worst])
        same = False

    # ---- Sample random points near center and filter polytope ones ----
    noise = torch.randn_like(pts_Z) * 0.5
    pts_rand = center[:, None, :] + noise

    mask = inside_polytope(A, b, pts_rand)
    pts_P = pts_rand[mask].view(center.shape[0], -1, center.shape[1])
    print(f"Sampled {pts_rand.shape[1]} random points, {pts_P.shape[1]} inside polytope.")

    # ---- Check polytope ⊆ zonotope ----
    ok2 = inside_zonotope(center, G, pts_P)
    viol = ~ok2
    if viol.any():
        num_viol = viol.sum().item()
        total = viol.numel()

        print(f"❌ {num_viol} / {total} polytope points are outside the zonotope "
            f"({100*num_viol/total:.2f}%)")
        b = 0  # batch 0
        idx = viol[b].nonzero()[0].item()
        x = pts_P[b, idx]

        print("❌ Polytope point outside zonotope:", x)

        # Solve for xi: x = c + G xi
        y = x - center[b]
        xi, *_ = torch.linalg.lstsq(G[b], y)

        recon = center[b] + G[b] @ xi

        print("Reconstruction error ||x - (c+Gxi)||:", torch.norm(x - recon))
        print("xi:", xi)
        print("max |xi|:", xi.abs().max())
        same=False
    
    if same:
        print("✅ Polytope and Zonotope represent the same set (up to sampling).")

def visualize_sets(center, G, A, b, num_samples=2000):
    """
    Visualize 2D zonotope and polytope.
    """
    B, d, k = G.shape
    assert d == 2, "Visualization only works for 2D."

    # Sample zonotope points
    pts_zono = sample_zonotope(center, G, n=num_samples)[0].cpu().numpy()  # batch 0

    # Sample random points and keep only those inside polytope
    noise = torch.randn_like(torch.tensor(pts_zono)) * 0.5
    pts_rand = center[0:1].cpu().numpy() + noise.numpy()
    x_tensor = torch.tensor(pts_rand, dtype=torch.float32, device=A.device).unsqueeze(0)  # shape: (1, n, d)
    inside_mask = inside_polytope(A, b, x_tensor).cpu().numpy()[0]  # batch 0
    pts_poly = pts_rand[inside_mask]

    # Plot sampled points
    plt.figure(figsize=(6,6))
    plt.scatter(pts_zono[:,0], pts_zono[:,1], color='blue', alpha=0.2, label='Zonotope samples')
    plt.scatter(pts_poly[:,0], pts_poly[:,1], color='green', alpha=0.2, label='Polytope samples')

    # Optionally plot convex hull of zonotope
    try:
        hull = ConvexHull(pts_zono)
        for simplex in hull.simplices:
            plt.plot(pts_zono[simplex, 0], pts_zono[simplex, 1], 'b-')
    except:
        pass

    # Optionally plot convex hull of polytope
    try:
        hull_p = ConvexHull(pts_poly)
        for simplex in hull_p.simplices:
            plt.plot(pts_poly[simplex,0], pts_poly[simplex,1], 'g-')
    except:
        pass

    plt.axis('equal')
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.title('Zonotope vs Polytope')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    from src.envs.navigate_seeker import NavigateSeekerEnv

    env = NavigateSeekerEnv(
        polytopic_approach=True,
        num_obstacles=1,
        min_radius=2.0,
        max_radius=4.0,
        draw_safe_action_set=False,
        num_envs=1,
        num_steps= 400,
    )
    env.reset()

    test_polytope_matches_zonotope(env)