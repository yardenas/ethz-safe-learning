import os
import yaml


def load_config_or_die(config_dir, config_basename):
    config = dict()
    for filename in ['model.yaml', 'policy.yaml']:
        filepath = os.path.join(config_dir, filename)
        with open(filepath, 'r') as file:
            config.update(yaml.safe_load(file))
    basepath = os.path.join(config_dir, config_basename)
    with open(basepath, 'r') as basefile:
        baseconfig = yaml.safe_load(basefile)
        update_values(baseconfig, config)
    return config


def update_values(update_from, update_to):
    for key in update_from:
        if key in update_to:
            if isinstance(update_from[key], dict) and isinstance(update_from[key], dict):
                update_values(update_from[key], update_to[key])
            else:
                update_to[key] = update_from[key]
        else:
            update_to[key] = update_from[key]
    return update_to



