import sys
import getopt
import time
import gym
import gym_reflected_xss
import uuid
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.a2c.policies import MlpPolicy, CnnPolicy
from stable_baselines3 import A2C
from stable_baselines3.common.env_util import make_vec_env
import torch as th
# remove tensorflow warning messages
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

def main(argv):
    start_url = ""
    test_suite_name = ""
    timesteps = 4000000
    deterministic = False
    deterministic_state = None
    deterministic_action = None

    try:
        opts, etc_args= getopt.getopt(argv[1:], "u:t:d:")
    except getopt.GetoptError:
        print("Use option -u")
        sys.exit(2)

    for opt,arg in opts:
        print(opt)
        if opt in ("-u"):
            option = arg
        if opt in ("-t"):
            timesteps = int(arg)
        if opt in ("-d"):
            # Backward compatibility: -d sets both state and action to deterministic
            if arg == 'a' or arg == 'action':
                deterministic_action = True
            elif arg == 's' or arg == 'state':
                deterministic_state = True
            else:
                deterministic = True
        """if opt == "-ds":
            # Set deterministic state space (requires argument: true/false)
            deterministic_state = True
        if opt  == "-da":
            # Set deterministic action space (requires argument: true/false)
            deterministic_action = True"""

    start_url = option
    learning_rate = 0.0005
    gamma = 0.95

    # Determine deterministic mode strings for logging
    if deterministic_state is None and deterministic_action is None:
        # Use old deterministic flag for both
        if deterministic:
            deterministic_mode = "d"
        else:
            deterministic_mode = "non-d"
    else:
        # Build mode string from separate flags
        state_mode = "ds" if (deterministic_state is not None) else "nds"
        action_mode = "da" if (deterministic_action is not None) else "nda"
        deterministic_mode = f"{state_mode}-{action_mode}"

    log_title = time.strftime('%Y.%m.%d-%H.%M.%S', time.localtime(time.time())) + "-" + test_suite_name +  "-" + str(timesteps) + "-A2C-learning-" + str(learning_rate) + "-gamma-" + str(gamma) + "O-" + str(deterministic_mode)


    print(deterministic_mode)
    print(deterministic, deterministic_state, deterministic_action )
    env = gym.make("reflected-xss-v0",
        start_url=start_url,
        mode=0,
        log_file_name=f"train-log-{log_title}.txt",
        deterministic=deterministic,
        deterministic_state=deterministic_state,
        deterministic_action=deterministic_action,
        max_steps_per_episode=500)

    # create learning agent
    print("[*] Creating A2C model ...")
    policy_kwargs = dict(activation_fn=th.nn.ReLU, net_arch=[dict(pi=[128,128,128], vf=[128,128,128])])


    model = A2C("MlpPolicy", env, verbose=1, tensorboard_log="./tensorboard_log/",
                learning_rate=learning_rate, gamma=gamma, policy_kwargs=policy_kwargs)
    print("[*] Start Agent learning ...")



    model.learn(total_timesteps=timesteps , tb_log_name=log_title)

    model_name = "models/" + log_title + "-" + str(uuid.uuid4()) + "-model.pkl"

    # save trained model
    model.save(model_name)

    # env.show_graph()

    del model

if __name__ == '__main__':
    main(sys.argv)

