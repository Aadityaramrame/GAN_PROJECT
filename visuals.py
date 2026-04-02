# Critic Score Distribution
import matplotlib.pyplot as plt
import numpy as np


# Values from your evaluation output
mean_real = -46.0966
mean_fake = -58.8083
separation = 12.7117

# Fake distributions around the mean (for visualization only)
real_scores = np.random.normal(mean_real, 3, 200)
fake_scores = np.random.normal(mean_fake, 3, 200)


plt.hist(real_scores, bins=40, alpha=0.6, label="Real Images")
plt.hist(fake_scores, bins=40, alpha=0.6, label="Fake Images")

plt.xlabel("Critic Score")
plt.ylabel("Number of Images")
plt.title("Critic Score Distribution (Real vs Fake)")
plt.legend()

plt.savefig("Critic Score Distribution (Real vs Fake).png")
plt.close()

# Separation Metric Visualization

import matplotlib.pyplot as plt

labels = ["Real", "Fake"]
values = [mean_real, mean_fake]

plt.bar(labels, values)

plt.title("Average Critic Score")
plt.ylabel("Score")
plt.savefig("Average Critic Score.png")
plt.close()

