import gym
import safety_gym


def main():
    sg6 = ['PointGoal1', 'PointGoal2', 'PointButton1',
           'PointPush1', 'CarGoal1', 'DoggoGoal1']
    for environment in sg6:
        env = gym.make('Safexp-' + environment + '-v0')
        env.reset()
        print("-------------------- " + environment + " --------------------\n",
              env.obs())


if __name__ == '__main__':
    main()
