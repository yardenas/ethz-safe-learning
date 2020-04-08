import os
import yaml


def load_config_or_die(congif_dir, config_basename):
    baseconfig = dict()
    for filename in ['model.yaml', 'policy.yaml']:
        filepath = os.path.join(congif_dir, filename)
        with open(filepath, 'r') as file:
            baseconfig.update(yaml.safe_load(file))
    basepath = os.path.join(congif_dir, config_basename)
    with open(basepath, 'r') as basefile:
        baseconfig.update(yaml.safe_load(basefile))
    return baseconfig

