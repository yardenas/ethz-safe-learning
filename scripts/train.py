import argparse
from config.config import load_config_or_die
from simba.infrastructure import RLTrainer
from simba.infrastructure.logger import logger
from simba.agents import MBRLAgent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--log_dir', type=str, default='/data/logs')
    parser.add_argument('--log_level', type=str, default='INFO')
    parser.add_argument('--config_dir', type=str, required=True)
    parser.add_argument('--config_basename', type=str, required=True)
    args = parser.parse_args()
    logger.setLevel(args.log_level)
    params = load_config_or_die(args.config_dir, args.config_basename)
    print(params)


if __name__ == '__main__':
    main()
