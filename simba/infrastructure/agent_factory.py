from simba.infrastructure.common import standardize_name
import simba.agents as agents


def make_agent(config, environment):
    agent = eval('agents.' + standardize_name(config['options']['agent']))
    agent_params = config['agents']['mbrl_agent']
    base_agent_params = config['agents']['agent']
    policy_params = config['policies'][agent_params['policy']]
    model_params = config['models'][agent_params['model']]
    return agent(seed=config['options']['seed'],
                 observation_space_dim=1,
                 action_space_dim=1,
                 policy_parameters=policy_params,
                 model_parameters=model_params,
                 **base_agent_params,
                 **agent_params)

