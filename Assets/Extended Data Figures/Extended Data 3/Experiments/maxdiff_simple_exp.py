from Maxdiff import SimpleMaxDiff
import numpy as np

def linear_model(s, u):
    return 1.0*s + 0.1 * u  # simple linear dynamics

def linear_model_true(s, u):
    return 1.05*s + 0.1 * u 

def state_cost(state,goal=0.0):
    return  1.0*(state-goal)**2 

agent = SimpleMaxDiff(model_fn=linear_model, cost_fn=state_cost, state_dim=1, action_dim=1, horizon=20,samples=50)

s0 = np.array([1.0])
M = 50
N_steps = 100
X = np.zeros((N_steps+1,M))
X[0,:] = s0
for j in range(M):
    for step in range(N_steps):
        u = agent.rollout(s0)
        # print(f"Step {step}, Action: {u}")
        s0 = linear_model_true(s0, u)
        print(f"New state {s0}, Action: {u}")
        X[step+1,j] = s0
        agent.shift_plan()

np.savez('1D_maxdiff_2050.npz', X=X)