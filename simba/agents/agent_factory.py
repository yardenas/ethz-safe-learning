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
        assert len(environment.action_space.shape) == 1 and \
               len(environment.observation_space.shape) == 1, "No support for non-flat action/observation spaces."
        kwargs = {**agent_params, **base_agent_params, 'policy_params': policy_params,
                  'model_params': model_params}
        return agents.MbrlAgent(seed=config['options']['seed'],
                                environment=environment,
                                **kwargs)
