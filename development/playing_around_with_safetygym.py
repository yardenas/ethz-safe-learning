import gym
import safety_gym
from simba.environment_utils.safety_gym_scoring import SafetyGymStateScorer


def main():
    sg6 = ['PointGoal1', 'PointGoal2', 'PointButton1',
           'PointPush1', 'CarGoal1', 'DoggoGoal1']
    for environment in sg6:
        env = gym.make('Safexp-' + environment + '-v0')
        env.reset()
        scorer = SafetyGymStateScorer(env.config)
        print("-------------------- " + environment + " --------------------\n",
              env.obs())


if __name__ == '__main__':
    main()
