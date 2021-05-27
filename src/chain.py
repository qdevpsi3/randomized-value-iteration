import dm_env
import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
from absl import app, flags
from bsuite.baselines import experiment
from bsuite.environments.base import Environment
from dm_env import specs

from rlsvi import RLSVI, Action, BasisFunction, Features, Horizon, Observation

FLAGS = flags.FLAGS
flags.DEFINE_integer("seed", 1234, "Random seed.")
flags.DEFINE_integer("size", 20, "Size of chain experiment.")
flags.DEFINE_bool("deterministic", False, "Deterministic environment.")
flags.DEFINE_integer("num_basis", 10, "Number of basis functions.")
flags.DEFINE_float("noise_std", 0.1, "Standard deviation of RLSVI.")
flags.DEFINE_float("reg_parameter", 1., "Regularization parameter of RLSVI.")
flags.DEFINE_integer("num_episodes", 200, "Number of train episodes.")


class ChainEnv(Environment):
    bsuite_num_episodes = 0

    def __init__(
        self,
        size: int = 10,
        deterministic: bool = False,
        seed: int = None,
    ):
        super().__init__()
        self._size = size
        self._deterministic = deterministic
        self._rng = np.random.RandomState(seed)
        if not self._deterministic:
            self._optimal_return = (1 - 1 / self._size)**(self._size - 1)
        else:
            self._optimal_return = 1.
        self._total_regret = 0
        self._reset()

    def _reset(self) -> dm_env.TimeStep:
        self._timestep = 0
        self._position = 0
        return dm_env.restart(self._get_observation())

    def _step(
        self,
        action: int,
    ) -> dm_env.TimeStep:
        if (action == 1) and (not self._deterministic):
            action = int(self._rng.rand() < 1. / self._size)
        self._timestep += 1
        self._position = max(
            0, min(self._position + 2 * action - 1, self._size - 1))
        observation = self._get_observation()
        reward = int(observation == self._size - 1)
        if self._timestep == self._size - 1:
            self._total_regret += self._optimal_return - reward
            self.bsuite_num_episodes += 1
            return dm_env.termination(reward=reward, observation=observation)
        return dm_env.transition(reward=reward, observation=observation)

    def _get_observation(self) -> Observation:
        obs = self._position
        return obs

    def observation_spec(self):
        return specs.DiscreteArray(self._size, name='position')

    def action_spec(self):
        return specs.DiscreteArray(2, name='action')

    def bsuite_info(self):
        return dict(total_regret=self._total_regret)


def random_coherent_basis(
    size: int,
    deterministic: bool,
    num_basis: int,
    rng: hk.PRNGSequence,
) -> BasisFunction:

    S = size
    A = 2
    H = S - 1
    K = num_basis
    if deterministic:
        p = 1.
    else:
        p = (1 - 1. / S)

    # optimal state value function
    aux = np.array([p**t for t in range(H, 0, -1)])
    aux = np.tril(jnp.ones(H)) @ np.diag(aux)
    aux = np.cumsum(aux[:, ::-1], axis=1)[:, ::-1]
    v_opt = np.concatenate([aux, aux[-1:, :]], axis=0).T

    # optimal state-action value function
    q_right = v_opt
    q_left = np.zeros((H, S))
    q_left[:-1, 1:] = v_opt[1:, :-1]
    q_opt = np.stack([q_left, q_right], axis=2)

    # coherent basis
    psi_ones = jnp.ones(shape=(H * S * A, 1))
    psi_opt = jnp.reshape(q_opt, [H * S * A, 1])
    psi_rand = jax.random.normal(next(rng), shape=(H * S * A, K - 2))
    psi = jnp.concatenate([psi_ones, psi_opt, psi_rand], axis=1)
    proj = psi @ jnp.linalg.inv(psi.T @ psi) @ psi.T
    w_ones = jnp.ones(shape=(H * S * A, 1))
    w_rand = jax.random.normal(next(rng), shape=(H * S * A, K - 1))
    w = jnp.concatenate([w_ones, w_rand], axis=1)
    w_proj = proj @ w
    w_proj /= jnp.linalg.norm(w_proj, axis=0)
    w_proj *= H * S * A
    phi = w_proj.reshape(H, S, A, K)

    # basis function
    def basis_function(
        h: Horizon,
        s: Observation,
        a: Action,
    ) -> Features:
        return phi[h, s, a].squeeze()

    return basis_function


def main(unused_arg):

    key = jax.random.PRNGKey(FLAGS.seed)
    rng = hk.PRNGSequence(key)

    # set environment
    env = ChainEnv(size=FLAGS.size,
                   deterministic=FLAGS.deterministic,
                   seed=FLAGS.seed)

    # set basis function
    basis_function = random_coherent_basis(size=FLAGS.size,
                                           deterministic=FLAGS.deterministic,
                                           num_basis=FLAGS.num_basis,
                                           rng=rng)

    # set agent
    agent = RLSVI(obs_spec=env.observation_spec(),
                  action_spec=env.action_spec(),
                  basis_function=basis_function,
                  sequence_length=FLAGS.size,
                  noise_std=FLAGS.noise_std,
                  reg_parameter=FLAGS.reg_parameter,
                  rng=rng)

    # run experiment
    experiment.run(agent=agent,
                   environment=env,
                   num_episodes=FLAGS.num_episodes,
                   verbose=True)


if __name__ == "__main__":
    app.run(main)
