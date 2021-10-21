# %%
term_state = np.array ([-0.03670745, -0.01818765, 0.22, -.1])
norm_state = np.array ([-0.03670745, -0.01818765, 0.01, -.1])
p = np.array([0.7, 0.3])
count0 = 0
count1 = 0
for i in range(1000):
    tmp = np.random.choice(np.array([0,1]), p = p)
    if tmp == 0:
        count0 += 1
    else: count1 += 1

print(count1)

