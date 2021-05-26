from copy import deepcopy
from functools import partial
from typing import Callable, List, NamedTuple, Tuple

import dm_env
import haiku as hk
import jax
from bsuite.baselines.base import Agent
from bsuite.baselines.utils import sequence
from dm_env import specs
from jax import numpy as jnp


class AgentState(NamedTuple):
    params_mean: jnp.ndarray
    params_cov: jnp.ndarray
    params_sample: jnp.ndarray


Action = int
BasisFunction = Callable[[jnp.ndarray, jnp.ndarray, jnp.ndarray], jnp.ndarray]


@partial(jax.jit, static_argnums=(2, ))
def _lstsq(
    A: jnp.ndarray,
    b: jnp.ndarray,
    reg_parameter: float,
) -> Tuple[jnp.ndarray, jnp.ndarray]:

    # get parameters covariance
    inv_cov = jnp.matmul(A.T, A) + reg_parameter * jnp.eye(A.shape[1])
    cov = jnp.linalg.inv(inv_cov)

    # get parameters mean
    mean = jnp.dot(jnp.matmul(cov, A.T), b)

    return mean, cov


def _rlsvi(
    agent_state: AgentState,
    basis_function: BasisFunction,
    trajectories: List[sequence.Trajectory],
    noise_std: float,
    reg_parameter: float,
    sequence_length: int,
    rng: hk.PRNGSequence,
) -> AgentState:
    basis_function = jax.vmap(basis_function)
    observations = jnp.stack([t.observations for t in trajectories])
    actions = jnp.stack([t.actions for t in trajectories])
    rewards = jnp.stack([t.rewards for t in trajectories])
    params_mean, params_cov, params_sample = [], [], []

    for h in range(sequence_length)[::-1]:

        # get regression matrix
        batch_h = h * jnp.ones(len(observations), dtype=int)
        batch_o = observations[:, h]
        batch_a = actions[:, h]
        A = basis_function(batch_h, batch_o, batch_a)
        A /= noise_std

        # get regression vector
        batch_r = rewards[:, h]
        batch_next_o = observations[:, h + 1]
        if h == sequence_length - 1:
            batch_next_r = rewards[:, -1]
        else:
            batch_next_q = basis_function(batch_h + 1, batch_next_o,
                                          None) @ params_sample[-1]
            batch_next_r = jnp.max(batch_next_q, axis=1)
        b = batch_r + batch_next_r
        b /= noise_std

        # get parameters
        mean, cov = _lstsq(A, b, reg_parameter)

        # get sample
        sample = jax.random.multivariate_normal(next(rng), mean, cov)

        params_mean.append(mean)
        params_cov.append(cov)
        params_sample.append(sample)

    return agent_state._replace(params_mean=params_mean[::-1],
                                params_cov=params_cov[::-1],
                                params_sample=params_sample[::-1])


class RLSVI(Agent):
    def __init__(
        self,
        obs_spec: specs.Array,
        action_spec: specs.DiscreteArray,
        basis_function: BasisFunction,
        sequence_length: int,
        noise_std: float,
        reg_parameter: float,
        rng: hk.PRNGSequence,
    ):

        self._obs_spec = obs_spec
        self._action_spec = action_spec
        self._sequence_length = sequence_length
        self._basis_function = basis_function
        self._noise_std = noise_std
        self._reg_parameter = reg_parameter
        self._rng = rng

        # initialize buffer
        self._trajectories = []
        self._buffer = sequence.Buffer(obs_spec, action_spec, sequence_length)

        # initialize parameters
        self._state = AgentState(None, None, None)

        self._rlsvi = _rlsvi

    def update(
        self,
        timestep: dm_env.TimeStep,
        action: Action,
        new_timestep: dm_env.TimeStep,
    ):
        self._buffer.append(timestep, action, new_timestep)
        if self._buffer.full() or new_timestep.last():
            trajectory = self._buffer.drain()
            self._trajectories.append(deepcopy(trajectory))
            self._state = self._rlsvi(self._state, self._basis_function,
                                      self._trajectories, self._noise_std,
                                      self._reg_parameter,
                                      self._sequence_length, self._rng)

    def select_action(
        self,
        timestep: dm_env.TimeStep,
    ) -> Action:
        if self._state.params_sample == None:
            action = jax.random.randint(next(self._rng), (), 0,
                                        self._action_spec.num_values)
            return int(action)
        h = int(timestep.step_type)
        o = timestep.observation
        basis = jnp.stack([
            self._basis_function(h, o, a)
            for a in range(self._action_spec.num_values)
        ],
                          axis=0)
        params = self._state.params_sample[h]
        action = jnp.argmax(basis @ params)
        return int(action)
