
        for m in range(num_episodes):
            A2C_net.train(env, gamma=gamma)
            if m % 100 == 0:
                print("Episode: {}".format(m))
                G = np.zeros(20)
                for k in range(20):
                    g = A2C_net.evaluate_policy(env)
                    G[k] = g

                reward_mean = G.mean()
                reward_sd = G.std()
                print("The test reward for episode {0} is {1} with sd of {2}.".format(m, reward_mean, reward_sd))
                reward_means.append(reward_mean)
        res[i] = np.array(reward_means)
    ks = np.arange(l)*100
    avs = np.mean(res, axis=0)
    maxs = np.max(res, axis=0)
    mins = np.min(res, axis=0)

    plt.fill_between(ks, mins, maxs, alpha=0.1)
    plt.plot(ks, avs, '-o', markersize=1)

    plt.xlabel('Episode', fontsize = 15)
    plt.ylabel('Return', fontsize = 15)

    if not os.path.exists('./plots'):
        os.mkdir('./plots')
        
    if A2C_net.type == 'A2C':
        plt.title("A2C Learning Curve for N = {}".format(args.n), fontsize = 24)
        plt.savefig("./plots/a2c_curve_N={}.png".format(args.n))
    elif A2C_net.type == 'Baseline':
        plt.title("Baseline Reinforce Learning Curve".format(args.n), fontsize = 24)
        plt.savefig("./plots/Baseline_Reinforce_curve.png".format(args.n))
    else: # Reinforce
        plt.title("Reinforce Learning Curve", fontsize = 24)
        plt.savefig("./plots/Reinforce_curve.png")
