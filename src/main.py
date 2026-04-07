from environment import GridWorldEnv

def main():
    env = GridWorldEnv(size=5)
    obs, _ = env.reset()
    done = False

    while not done:
        action = env.action_space.sample()  # Random action
        obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        env.render()

if __name__ == "__main__":
    main()