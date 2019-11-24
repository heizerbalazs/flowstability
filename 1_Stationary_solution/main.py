from velocity_profile import VelocityProfileInitializer

import matplotlib.pyplot as plt
import numpy as np

velocityProfile = VelocityProfileInitializer.initialize('Blasius')
x_start, x_end = velocityProfile.domain
x = np.linspace(x_start,x_end,1000)

fig1 = plt.figure()
ax1 = fig1.add_subplot(1,1,1)
ax1.plot(velocityProfile.U(x),x)
plt.show()


