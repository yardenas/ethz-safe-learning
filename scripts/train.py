def main():
    import argparse
    import logging
    import time
    from config.config import load_config_or_die, pretty_print
    from simba.infrastructure.logging_utils import init_loggging
    from simba.infrastructure import RLTrainer
    from simba.infrastructure.agent_factory import make_agent
    parser = argparse.ArgumentParser()
    now = time.asctime(time.localtime())
    parser.add_argument('--log_dir', type=str, default=('/data/logs' + now))
    parser.add_argument('--log_level', type=str, default='INFO')
    parser.add_argument('--config_dir', type=str, required=True)
    parser.add_argument('--config_basename', type=str, required=True)
    args = parser.parse_args()
    init_loggging(args.log_level)
    params = load_config_or_die(args.config_dir, args.config_basename)
    logging.info("Startning a training session with parameters:\n" +
                 pretty_print(params))
    make_agent(params, None)


if __name__ == '__main__':
    main()
