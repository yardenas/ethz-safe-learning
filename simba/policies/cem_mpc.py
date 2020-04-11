# def _compile_cost(self, ac_seqs, get_pred_trajs=False):
#     t, nopt = tf.constant(0), tf.shape(ac_seqs)[0]
#     init_costs = tf.zeros([nopt, self.npart])
#     ac_seqs = tf.reshape(ac_seqs, [-1, self.plan_hor, self.dU])
#     ac_seqs = tf.reshape(tf.tile(
#         tf.transpose(ac_seqs, [1, 0, 2])[:, :, None],
#         [1, 1, self.npart, 1]
#     ), [self.plan_hor, -1, self.dU])
#     init_obs = tf.tile(self.sy_cur_obs[None], [nopt * self.npart, 1])
#
#     def continue_prediction(t, *args):
#         return tf.less(t, self.plan_hor)
#
#     if get_pred_trajs:
#         pred_trajs = init_obs[None]
#
#         def iteration(t, total_cost, cur_obs, pred_trajs):
#             cur_acs = ac_seqs[t]
#             next_obs = self._predict_next_obs(cur_obs, cur_acs)
#             delta_cost = tf.reshape(
#                 self.obs_cost_fn(next_obs) + self.ac_cost_fn(cur_acs), [-1, self.npart]
#             )
#             next_obs = self.obs_postproc2(next_obs)
#             pred_trajs = tf.concat([pred_trajs, next_obs[None]], axis=0)
#             return t + 1, total_cost + delta_cost, next_obs, pred_trajs
#
#         _, costs, _, pred_trajs = tf.while_loop(
#             cond=continue_prediction, body=iteration, loop_vars=[t, init_costs, init_obs, pred_trajs],
#             shape_invariants=[
#                 t.get_shape(), init_costs.get_shape(), init_obs.get_shape(), tf.TensorShape([None, None, self.dO])
#             ]
#         )
#
#         # Replace nan costs with very high cost
#         costs = tf.reduce_mean(tf.where(tf.is_nan(costs), 1e6 * tf.ones_like(costs), costs), axis=1)
#         pred_trajs = tf.reshape(pred_trajs, [self.plan_hor + 1, -1, self.npart, self.dO])
#         return costs, pred_trajs
#     else:
#         def iteration(t, total_cost, cur_obs):
#             cur_acs = ac_seqs[t]
#             next_obs = self._predict_next_obs(cur_obs, cur_acs)
#             delta_cost = tf.reshape(
#                 self.obs_cost_fn(next_obs) + self.ac_cost_fn(cur_acs), [-1, self.npart]
#             )
#             return t + 1, total_cost + delta_cost, self.obs_postproc2(next_obs)
#
#         _, costs, _ = tf.while_loop(
#             cond=continue_prediction, body=iteration, loop_vars=[t, init_costs, init_obs]
#         )
#
#         # Replace nan costs with very high cost
#         return tf.reduce_mean(tf.where(tf.is_nan(costs), 1e6 * tf.ones_like(costs), costs), axis=1)
from simba.infrastructure.logger import logger
from simba.policies.policy import PolicyBase


class CemMpc(PolicyBase):
    def __init__(self, transition_model):
        super().__init__()
        self.transition_model = transition_model
        pass

    def generate_action(self, state):
        logger.debug("Taking action.")
        pass

    def build(self):
        logger.debug("Building policy.")
        pass
