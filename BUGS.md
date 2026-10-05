# What went wrong in aur_ppo's PPO

A post-mortem on why equivariant PPO in this repo couldn't learn from sparse
rewards. Every code quote is from `main` at `863dd8a`, before the fixes. The
fixes are in commit `490cd6c` on branch `fix/ppo-sparse-reward-bugs`.

## Summary

Sparse-reward manipulation depends on two things: **demonstrations** to get the
policy near success, and **credit assignment** to propagate the rare +1 back to
the actions that earned it. In this code both were broken, and nothing crashed:

| # | Bug | Where | Effect |
|---|-----|-------|--------|
| 1 | BC pretraining loss had no gradient path to the actor | `robot_ppo.py` | Pretraining did nothing |
| 2 | Expert term in the PPO loss had no gradient path | `robot_ppo.py` | Demos never shaped PPO |
| 3 | Log-prob buffer had the wrong shape | `robot_ppo.py` | PPO ratio compared unrelated samples |
| 4 | Envs were never reset during rollouts | `robot_ppo.py` | Every step looked terminal, so no credit assignment |
| 5 | GAE skipped the last step and never bootstrapped | `robot_ppo.py` | Truncated rollouts lost their tail |
| 6 | Pretraining used 8 of ~5000 frames | `robot_ppo.py` | (moot, given bug 1) |
| 7 | `type=bool` CLI flags | `robot_run.py` | `--equivariant False` enabled equivariance |
| 8 | Import broken by a refactor | `robot_ppo.py` | The entry point couldn't run since May 2024 |
| 9 | Gradient clipping skipped the critic | `robot_ppo.py` | Minor |
| 10 | GAE body outside its loop | `ppoBullet.py` | Only one advantage row computed |
| 11 | "Not done" masks treated as dones | `ppoBullet.py` | Episode boundaries inverted |
| 12 | Ratio used a freshly sampled action | `ppoBullet.py` | Not PPO |
| 13 | Advantages indexed twice | `ppoBullet.py` | Wrong advantages per sample |
| 14 | SAC log-prob evaluated at the wrong point | `nets/nets.py` | Wrong entropy term in SAC/AWAC |
| 15 | Equivariant actor's std not equivariant | `nets/equiv.py` | Symmetry silently violated |

Bugs 1–4 are each, on their own, enough to stop a sparse-reward agent from
learning.

The sections below group the bugs by the *kind* of mistake. The kind matters
more than the individual line, because it tells you what to check next time.

---

## A. Gradient flow: losses that couldn't train anything

### Bug 1 — BC pretraining was a no-op

`src/robot_ppo.py:301`
```python
expert_loss = nn.functional.mse_loss(b_actions[mb_inds].requires_grad_(True), b_true_actions[mb_inds])
```

**What happened.** `b_actions` came from the buffer, filled during collection:

```python
with torch.no_grad():                                              # line 202
    scaled_agent, unscaled_agent, logprob, _, value = self.policy.evaluate(...)
    self.pretrain_buffer.actions[step] = unscaled_agent            # line 209
```

Under `no_grad`, autograd records nothing. The stored numbers have no link back
to the weights that produced them. `b_actions[mb_inds]` (indexing with an
array) makes yet another fresh copy. `requires_grad_(True)` turns that copy
into a new *leaf* tensor, so `backward()` computes a gradient for that
temporary copy and throws it away. The actor's parameters get no gradient, and
`optimizer.step()` leaves them unchanged. The phase ran, printed "Pretraining
complete", and left the policy as random as it started.

**Why the `requires_grad_` was there.** Probably to silence this error:
`element 0 of tensors does not require grad and does not have a grad_fn`.
That error was the real signal that the loss wasn't connected to the network.
`requires_grad_` makes the error go away without connecting anything.

**Would removing `no_grad` have fixed it?** No. Tested directly:
- **It crashes on the second minibatch.** Writing graph-carrying tensors into a
  preallocated buffer chains every write together (`CopySlices`), so the second
  `backward()` fails with "Trying to backward through the graph a second time".
- **Memory.** Keeping the graph alive holds every activation of every rollout
  forward pass.
- **Even with infinite memory, the gradient is stale.** Stored actions were
  computed by *old* weights, and they don't change after `optimizer.step()`.
- **They're samples, not the mean.** They're mean plus exploration noise, so
  MSE through them also pushes the std toward zero.

**Fix.** Store observations, and recompute the policy's output inside the loss:
```python
action_mean = self.policy.action_mean(b_states[mb_inds], b_obs[mb_inds])  # fresh forward pass
expert_loss = nn.functional.mse_loss(action_mean, b_true_actions[mb_inds])
```

### Bug 2 — The expert term in the PPO loss did nothing

`src/robot_ppo.py:397`
```python
expert_loss = nn.functional.mse_loss(b_actions[mb_inds], b_true_actions[mb_inds])
loss = policy_loss - self.entropy_coeff * entropy_loss + value_loss + self.expert_weight * expert_loss
```

This is the same mistake without the workaround. Both tensors are constants.
`backward()` doesn't complain this time, because the policy and value terms
*do* have graphs. But this term contributes zero gradient. With
`expert_weight=0.9` it looked like the demos were heavily weighted. In fact
they had no effect. Fixed by using the policy mean from the forward pass the
update already does.

> **Rule.** The buffer holds *data*: observations, actions taken, expert labels,
> rewards, and old log-probs and old values (things meant to stay fixed).
> Anything that should change as the weights change (means, log-probs, values)
> is recomputed inside the loss. Quick test: if a tensor in your loss came out
> of the buffer, the optimizer can't move it. That's correct for targets and
> baselines, and a bug for the thing you're training.

---

## B. Data layout: shapes and indices

### Bug 3 — The log-prob buffer had the wrong shape

`src/robot_ppo.py:31, 169, 60`
```python
self.log_probs = torch.zeros((num_steps, num_envs, action_shape))   # [T, N, 5]
...
self.buffer.log_probs[step] = logprob     # logprob is [N]: already summed over action dims
...
b_logprobs = self.log_probs.reshape(-1)   # T*N*5 entries; every other flattened tensor has T*N
```

**What happened.** Assigning a `[5]` vector into a `[5, 5]` slot broadcasts
along the *last* axis, so every row gets the full vector of all five envs'
log-probs. After flattening, entry `k` holds the log-prob of env `k % 5` at
step `k // 25` instead of step `k // 5`. Every `b_logprobs[mb_inds]` lookup
therefore came from the first fifth of the rollout. The PPO ratio
π_new(a)/π_old(a) compared an action with the old log-prob of a *different*
action, so clipping and the policy gradient were essentially noise.

**Why it never crashed.** Pure coincidence: the default `num_envs` is 5, and
so is the action dimension. Broadcasting `[N]` into `[N, 5]` only works when
`N == 5` (or 1). With 4 envs it fails immediately ("The expanded size of the
tensor (5) must match the existing size (4)"). A shape coincidence hid the
bug.

**Fix.** `torch.zeros((num_steps, num_envs))`.

> **Rule.** Assert shapes where buffers are written:
> `assert logprob.shape == (num_envs,)`. Avoid making unrelated dimensions
> equal in debug configs. Test with `num_envs=3` precisely *because* nothing
> else is 3.

### Bug 6 — Pretraining used only 8 frames per epoch

`src/robot_ppo.py:441, 293, 300`
```python
self.pretrain_update(flattened_pretrain_buffer, self.num_update_epochs, self.pretrain_batch_size, self.pretrain_minibatch_size)
...
b_inds = np.arange(batch_size)                          # batch_size = pretrain_batch_size = 8
...
mb_inds = b_inds[index:index+self.minibatch_size]       # PPO's minibatch size, not the argument
```

The function's `batch_size` was meant to be "number of collected frames"
(5 envs × 1000 steps = 5000), but it received the CLI's `pretrain_batch_size`
(8). Only frames 0–7 were ever used. This was moot only because bug 1 made
pretraining a no-op anyway.

**Fix.** Pass `self.total_pretrain_steps`, and use the `minibatch_size` argument.

### Bug 13 — Advantages indexed twice (`ppoBullet.py`)

`src/policies/ppoBullet.py:118, 191`
```python
advantages = self.loss_calc_dict['advantages'][inds]    # inside get_buffer_values(mb_inds)
...
mb_advantages = advantages[mb_inds]                      # indexes the already-indexed tensor
```

This takes `advantages[mb_inds][mb_inds]`. The minibatches aren't shuffled
here, so the first one (`mb_inds = 0..m-1`) happens to come out right. The
second one indexes an `m`-element tensor with values `m..2m-1` and raises an
`IndexError`. With shuffled indices it would silently give each sample
someone else's advantage.

---

## C. RL semantics: episodes, bootstrapping, and the PPO ratio

### Bug 4 — Rollouts never reset finished environments

`src/robot_ppo.py:177`
```python
next_states, next_obs, reward, done = self.envs.step(actions)
```

`EnvWrapper.step` defaults to `auto_reset=False`, and BulletArm never resets
on its own. In `close_loop_env.py` the step function sets
`done = current_episode_steps >= max_steps`, which stays `True` once an
episode hits 100 steps. `train()` resets only once, before the first update.
So after 100 steps, **every env reports `done=True` on every subsequent
step**. GAE then treats each transition as the last of its episode. Advantages
collapse to one-step returns, and a sparse +1 can never be credited to the
actions that led up to it.

The pretraining and test code both passed `auto_reset=True`. Only the main
rollout didn't.

**Fix.** `self.envs.step(actions, auto_reset=True)`.

> **Rule.** Log episode lengths and look at them. Here they'd have shown a
> wall of length-1 "episodes" after the first 100 steps.

### Bug 5 — GAE skipped the last step and never bootstrapped

`src/robot_ppo.py:230–231`
```python
for t in reversed(range(num_steps - 1)):      # never visits t = num_steps - 1
    if t == self.num_steps - 1:               # so this branch never runs
        nextnonterminal = 1.0 - next_done
        nextvalues = next_value
```

The value of the state after the rollout (`next_value`) was never used, and
the last step's advantage stayed at 0. A rollout of 1024 steps cuts episodes
mid-way, and bootstrapping from V(s_T) is how GAE accounts for the reward
still to come.

**Fix.** `range(num_steps)` and `t == num_steps - 1`.

### Bug 10 — The GAE body sat outside its loop (`ppoBullet.py`)

`src/policies/ppoBullet.py`, `run_gae`
```python
for t in reversed(range(batch_size)):
    if t == batch_size - 1:
        ...
    else:
        ...
# <- loop ends here
delta = rewards[t] + self.gamma * nextvalues * nextnonterminal - values[t]
advantages[t] = lastgaelam = delta + self.gamma * self.gae_lambda * nextnonterminal * lastgaelam
```

The `delta`/`advantages[t]` lines are dedented, so they run once, after the
loop, with `t = 0`. Every other advantage stays 0. Python's whitespace makes
this an easy slip and an invisible one, since the code is syntactically fine.

### Bug 11 — "Not done" masks treated as dones (`ppoBullet.py`)

```python
non_final_masks = (dones_tensor ^ 1).float()        # 1 = episode CONTINUES
...
batch_size, states, obs, actions, rewards, dones, ... = self._loadLossCalcDict()
#                                         ^^^^^ actually non_final_masks
nextnonterminal = 1.0 - dones[t + 1]                # = done, i.e. inverted
```

The tuple returned `non_final_masks` in the slot that `run_gae` named
`dones`, so `1 - dones` flipped every episode boundary. Here dones are stored
per transition (done after step `t`), so the index should also be `t`, not
`t + 1`.

> **Rule.** Positional tuples of 10 values invite this. Return a dict or a
> dataclass, and name things by meaning (`not_done`, `episode_continues`).

### Bug 12 — The PPO ratio used a freshly sampled action (`ppoBullet.py`)

`src/policies/ppoBullet.py:184`
```python
a, newlogprob, mean, entropy = self.pi.sample(obs.cuda())
log_ratio = newlogprob - old_log_probs
```

PPO's ratio is π_new(**a_t**) / π_old(**a_t**) for the action **actually
taken** at collection time. `sample(obs)` draws a *new* action and returns
its log-prob, so the ratio compares two different actions and is meaningless.
The expert MSE on that fresh sample did have a gradient, but it also pushed
the std toward zero.

**Fix.** Evaluate the stored action (pass its pre-tanh value to `sample`),
and regress the policy *mean* onto the expert.

---

## D. Probability: densities and symmetry

### Bug 14 — SAC log-prob evaluated at the wrong point

`src/nets/nets.py:77–81`, `SACGaussianPolicyBase.sample` (siege has the same
code, also fixed)
```python
y_t = torch.tanh(x_t)
x_t = y_t                                # overwrites the pre-tanh sample
log_prob = normal.log_prob(x_t)          # Gaussian density at tanh(x), not at x
log_prob -= torch.log((1 - y_t.pow(2)) + epsilon)
```

For a tanh-squashed Gaussian, change of variables gives
log π(y) = log N(**x**; μ, σ) − Σ log(1 − tanh(x)²). The code evaluated the
Gaussian at `tanh(x)` instead. Near 0 the two agree, which is why it looks
fine. Near ±1 they diverge badly. With μ = 2, σ = 0.1, x = 2 (action ≈ 0.96):

- correct Gaussian term: ≈ +1.4
- buggy Gaussian term: ≈ −52

So confident, saturated actions (gripper commands, full-speed moves) look
enormously random. That corrupts SAC's entropy bonus, the critic's soft
target, and automatic α tuning, for every SAC/AWAC agent built on this class.
The fix is one line: compute `log_prob` *before* reassigning `x_t`.

### Bug 15 — The equivariant actor's std wasn't equivariant

`src/nets/equiv.py:89, 123` (inherited from the Equivariant SAC design)
```python
log_std = conv_out[:, self.action_dim:]   # action_dim invariant (trivial) outputs
```

The mean of (dx, dy) correctly comes from a rotating irrep(1) field. But the
two stds are separate *invariant* scalars. A 90° rotation maps (dx, dy) to
(−dy, dx). An equivariant Gaussian must therefore swap σx and σy, yet
invariant outputs can't swap. The distribution is only equivariant when
σx = σy.

**Fix.** dx and dy share one std (their mean). siege's new
`nets/equivariant/` builds this in, and has tests that check the whole
distribution transforms correctly.

---

## E. Python and tooling

### Bug 7 — `type=bool` in argparse

`src/robot_run.py` (8 flags, including `--equivariant`, `--gae`, `--norm_adv`)
```python
parser.add_argument('-eq', '--equivariant', type=bool, ...)
```

argparse calls `bool("False")`, which is `True`, because any non-empty string
is truthy. So `--equivariant False` *enabled* equivariance, and no boolean
flag could be turned off from the command line. The file already had a
correct `str2bool`; it just wasn't used for these flags.

### Bug 8 — An import broken by a refactor

`src/robot_ppo.py:7`
```python
from models import robot_actor_critic          # imports the MODULE
self.policy = robot_actor_critic(device, ...)  # TypeError: 'module' object is not callable
```

Commit `7205b1f` ("centralizing", May 2024) moved the class into
`models/robot_actor_critic.py` without updating this import. From then on,
the README's main command couldn't run. A one-line smoke test (construct the
agent, run one update) catches this kind of break the moment it happens.

### Bug 9 — Gradient clipping skipped the critic

```python
nn.utils.clip_grad_norm_(self.policy.actor.parameters(), self.max_grad_norm)
```

One optimizer updates both actor and critic, but only the actor's gradients
were clipped. This is minor, but value-loss spikes early in sparse-reward
training are exactly what clipping is for.

### Siege (same kinds of mistakes)

- `policies/ppo.py` was an empty stub, and `trainer/ppoTrainer.py` mixed tabs
  and spaces, so the module couldn't even be imported. The PPO path had never
  run.
- `nets/sense_2.py` and `policies/ppo_dep.py`: a find-and-replace mangled
  `device = torch.device("cpu")` into
  `from siege.src.utils.misc import devicetorch.device("cpu")`, which is a
  syntax error.
- `nets/sense_2.py:185`: `if special_policy is 'awac'` compares object
  identity, not string value. It only works when CPython happens to intern
  both strings. Use `==`.

---

## What would have caught all of this

Every bug above was **silent**. RL is uniquely bad at hiding mistakes,
because a broken agent and a correct agent on a hard sparse-reward task look
identical: return stays at 0 either way. You can't use "does it learn?" as
your test. These checks are cheap and catch everything in this list:

1. **Overfit BC on 32 samples.** The loss must go to ~0. This catches bugs 1,
   2 and 6.
2. **The PPO ratio is exactly 1 on the first minibatch** of the first epoch,
   before any update. This catches bugs 3 and 12.
3. **Parameters change after an update.** Snapshot them before, compare after.
   This catches bugs 1 and 2.
4. **Assert shapes where buffers are written.** Test with a deliberately odd
   `num_envs`. This catches bugs 3 and 13.
5. **Check GAE against a brute-force implementation** on a tiny random rollout.
   This catches bugs 5, 10 and 11.
6. **Log episode lengths and success rate, and look at them.** This catches
   bug 4.
7. **Solve a toy first.** A 2-D sparse-reward reach task trains in minutes. If
   that doesn't learn, BulletArm won't either.
8. **Test symmetry properties numerically.** Rotate the input, check the output
   transforms. This catches bug 15.
9. **Run the entry point in CI** for one update. This catches bugs 7 and 8.

Most of these now exist as tests in siege (`tests/test_ppo.py`,
`tests/test_envs_and_gae.py`, `tests/test_equivariance.py`). The habit worth
keeping: **assume new RL code is wrong until a test says otherwise.**
