import numpy as np
import matplotlib.pylab as plt
from matplotlib import cm
from matplotlib import animation, rc
from IPython import display
from IPython.display import HTML, clear_output

# Heun
def Heun(x, f, y0): # Variable y' = f(x,y), Condición inicial
    n = len(x)
    a = x[0]
    b = x[-1]
    h = (b-a)/n
    m = len(y0)
    y = np.zeros((m,n))
    y[:,0] = y0
    for i in range(n-1):
        clist = []
        dlist = []
        for k in range(m):
            clist.append(f(x[i], y[:,i])[k])
            dlist.append(f(x[i] + (2/3)*h, y[:,i] + (2/3)*h*(f(x[i], y[:,i])[k]))[k])
        c = np.array(clist)
        d = np.array(dlist)
        y[:,i+1] = y[:,i] + (h/4)*(c + 3*d)   
    return y
	
	

class doblepend:
  def __init__(self, m1, m2, g, l1, l2):
    self.m1 = m1
    self.m2 = m2
    self.g = g
    self.l1 = l1
    self.l2 = l2
  def __call__(self, x, y0):
    def f(x, y):
      tp10 = y[1]
      tp11 = (-self.g*(2*self.m1 + self.m2)*np.sin(y[0]) - self.m2*self.g*np.sin(y[0] - 2*y[2]) - 2 * np.sin(y[0] - y[2])*self.m2*(y[3]**2 * self.l2 + y[1]**2 * self.l1 * np.cos(y[0] - y[2])))/(self.l1*(2*self.m1 + self.m2 - self.m2*np.cos(2*y[0] - 2*y[2])))
      tp22 = y[3]
      tp23 = (2 * np.sin(y[0] - y[2]) * (y[1]**2 * self.l1*(self.m1+self.m2) + self.g*(self.m1 + self.m2)*np.cos(y[0]) + y[3]**2 * self.l2 * self.m2 * np.cos(y[0] - y[2])))/(self.l2*(2*self.m1 + self.m2 - self.m2*np.cos(2*y[0] - 2*y[2])))
      return [tp10, tp11, tp22, tp23]
    
    the1 = Heun(x, f, y0)[0]
    thep1 = Heun(x, f, y0)[1]
    the2 = Heun(x, f, y0)[2]
    thep2 = Heun(x, f, y0)[3]
    tr = np.zeros((4, len(x)))
    x1 = self.l1*np.sin(the1)
    y1 = -self.l1*np.cos(the1)
    x2 = x1 + self.l2*np.sin(the2)
    y2 = y1 - self.l2*np.cos(the2)
    tr[0] = x1
    tr[1] = y1
    tr[2] = x2
    tr[3] = y2
    return tr

t = np.linspace(0, 4, 1000)
y0d = np.array([1.7, 0.5, 1.3, 1])

# Péndulo doble
m1 = 1
m2 = 1
l1 = 1
l2 = 1
g = 9.81


d = doblepend(m1, m2, g, l1, l2)
trpend = d(t, y0d)

fig, ax = plt.subplots()
lineaun1, = ax.plot([], [], 'k-', lw=2)
lineaun2, = ax.plot([], [], 'k-', lw=2)
linea, = ax.plot([], [], 'r-', lw=0.5, alpha = 0.4)
punto, = ax.plot([], [], 'ko')
punto1, = ax.plot([], [], 'ko')
ax.set_xlim(-l1-l2 -1, l1+l2 + 1)
ax.set_ylim(-l1-l2 -1, l1+l2 + 1)

'''
def graficar(t, f):
  lineaun1.set_data([0, f[0,t]], [0, f[1,t]])
  lineaun2.set_data([f[0,t], f[2,t]], [f[1,t], f[3,t]])
  linea.set_data(f[2,:t], f[3, :t])
  punto.set_data(f[0,t], f[1,t])
  punto1.set_data(f[2,t], f[3,t])

anim = animation.FuncAnimation(fig, graficar, interval=max(t), frames=len(t), fargs=[trpend],repeat=False)



plt.close()
HTML(anim.to_html5_video())

'''

plt.plot(trpend[0], trpend[1])
plt.plot(trpend[2], trpend[3])


plt.show()








