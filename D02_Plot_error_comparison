import matplotlib as plt
from D02 Plot_CrossCorrelation.py import Plot_CC
curves_CC = Plot_CC()
fig, ax = plt.subplots(1,1)  
for j in range(2):
    for i in range(2):
        ax[j,i].plot(curves_CC[i+j])
        ax[j,i].plot(curves_OF[i+j])
        ax[j,i].set_xlabel("x")
        ax[j,i].set_ylabel(r'Normalized $\frac{d\rho}{dx}$')
        ax[j,i].set_title("OF vs XCOR normalized density gradient"+ str(i+j))

fig.tight_layout()
plt.show()