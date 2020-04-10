import os
import yaml


def load_config_or_die(config_dir, config_basename):
    config = dict()
    for filename in ['models.yaml', 'policies.yaml', 'agents.yaml']:
        filepath = os.path.join(config_dir, filename)
        with open(filepath, 'r') as file:
            config.update(yaml.safe_load(file))
    basepath = os.path.join(config_dir, config_basename)
    with open(basepath, 'r') as basefile:
        baseconfig = yaml.safe_load(basefile)
        overwrite_default_values(baseconfig, config)
    return config


def pretty_print(config, indent=0):
    summary = str()
    align = 30 - indent * 2
    for key, value in config.items():
        summary += '  ' * indent + '{:{align}}'.format(str(key), align=align)
        if isinstance(value, dict):
            summary += '\n' + pretty_print(value, indent + 1)
        else:
            summary += '{}\n'.format(str(value))
    return summary


def overwrite_default_values(update_from, update_to):
    for key in update_from:
        if key in update_to:
            if isinstance(update_from[key], dict) and isinstance(update_from[key], dict):
                overwrite_default_values(update_from[key], update_to[key])
            else:
                update_to[key] = update_from[key]
        else:
            update_to[key] = update_from[key]
    return update_to



