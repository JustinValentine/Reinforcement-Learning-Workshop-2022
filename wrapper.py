import gym 

from ale_py import ALEInterface
ale = ALEInterface()

from ale_py.roms import Breakout
ale.loadROM(Breakout)

import cv2

import time 

env = gym.make("ALE/Breakout-v5")
env.reset()

step_num, total_reward = 0, 0
while True:
    action = env.env.action_space.sample()
    state, reward, done, _, _ = env.step(action)
    print(f"S{step_num}: {action}, {reward}, {total_reward}")
    if done:
        break

    img = cv2.resize(state, (480, 480), 
                     interpolation=cv2.INTER_CUBIC)
    cv2.imshow('Frame', img)
    key_code = cv2.waitKey(1)
    if key_code & 0xFF == 27:
        break
    time.sleep(0.01)

    step_num += 1

env.close()