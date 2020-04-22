from simba.infrastructure.common import standardize_name
import simba.agents as agents


def make_agent(config, environment):
    agent_name = config['options']['agent']
    assert agent_name in config['agents'], "Specified agent does not exist."
    agent = eval('agents.' + standardize_name(agent_name))
    agent_params = config['agents'][agent_name]
    base_agent_params = config['agents']['agent']
    policy = agent_params['policy']
    assert policy in config['policies'], "Specified policy does not exist."
    policy_params = config['policies'][policy]
    model = agent_params['model']
    assert model in config['models'], "Specified model does not exist."
    model_params = config['models'][model]
    if agent is agents.MbrlAgent:
        policy_params['environment'] = environment
        kwargs = {**agent_params, **base_agent_params, 'policy_params': policy_params,
                  'model_params': model_params}
        return agents.MbrlAgent(seed=config['options']['seed'],
                                observation_space_dim=environment.observation_space.shape[0],
                                action_space_dim=environment.action_space.shape[0],
                                **kwargs)

