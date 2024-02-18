# dsp
Sin Wave(continuous): 
import math 
import numpy as np 
import matplotlib.pyplot as plt 
def sine_signal_continous(): 
 x = np.arange(0, 20) 
 y = np.sin(x) 
 plt.figure(figsize=(5,5)) 
 plt.xlabel("t in secs", fontweight="bold") 
 plt.ylabel('sin(t)', fontweight="bold") 
 plt.title('Sine Function signal Continuous', 
fontweight="bold") 
 plt.plot(x,y, marker="o") 
 plt.show() 
sine_signal_continous() 

Sin Wave(descrete):
import math
import numpy as np
import matplotlib.pyplot as plt
def sine_signal_discrete(): 
 x = np.arange(0, 20)
 y = np.sin(x) 
 plt.figure(figsize=(5,5))
 plt.xlabel("t in secs", fontweight="bold") 
 plt.ylabel('sin(t)', fontweight="bold")
 plt.title('Sine Function signal discrete', fontweight="bold") 
 plt.stem(x,y)
 plt.show() 
sine_signal_discrete()

Cos Wave(continuous): 
import math 
import numpy as np 
import matplotlib.pyplot as plt 
def cos_signal_continous(): 
 x = np.arange(0, 20) 
 y = np.cos(x) 
 plt.figure(figsize=(5,5)) 
 plt.xlabel("t in secs", fontweight="bold") 
 plt.ylabel('cos(t)', fontweight="bold") 
 plt.title('cos Function signal discrete', fontweight="bold") 
 plt.plot(x,y,marker='o') 
 plt.show() 
cos_signal_continous()

Cos Wave(discrete):
import math
import numpy as np
import matplotlib.pyplot as plt
def cos_signal_discrete(): 
 x = np.arange(0, 20)
 y = np.cos(x) 
 plt.figure(figsize=(5,5))
 plt.xlabel("t in secs", fontweight="bold") 
 plt.ylabel('cos(t)', fontweight="bold")
 plt.title('cos Function signal discrete', fontweight="bold") 
 plt.stem(x,y)
 plt.show() 
cos_signal_discrete()

Square Wave(continuous): 
import numpy as np 
import matplotlib.pyplot as plt 
def square_wave(t, period, duty_cycle): 
 frequency = 1 / period 
 phase_shift = duty_cycle * period 
 return np.where((t % period) < phase_shift, 1, -1) 
t = np.linspace(0, 2, 1000) 
period = 1 
duty_cycle = 0.5 
square_wave_signal = square_wave(t, period, duty_cycle) 
plt.plot(t, square_wave_signal) 
plt.title('Continuous Square Wave') 
plt.xlabel('Time (s)') 
plt.ylabel('Amplitude') 
plt.grid(True) 
plt.show()

Square Wave(discrete):
import numpy as np
import matplotlib.pyplot as plt
def square_wave(t, period, duty_cycle):
 frequency = 1 / period
 phase_shift = duty_cycle * period
 return np.where((t % period) < phase_shift, 1, -1)
t = np.linspace(0, 2, 10) 
period = 1 
duty_cycle = 0.5 
square_wave_signal = square_wave(t, period, duty_cycle)
plt.stem(t, square_wave_signal)
plt.title('Discrete Square Wave')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.grid(True) 
plt.show()

Triangular Wave(continuous):
import numpy as np 
import matplotlib.pyplot as plt 
def triangle_signal_contn(): 
 x = range(0, 10) 
 y = [] 
 for i in x: 
 y.append(3*abs(((i-1) % 4) - 2) - 3) 
 plt.figure(figsize=(4,4)) 
 plt.xlabel("t in secs", fontweight="bold") 
 plt.ylabel("T(t)", fontweight="bold") 
 plt.title("Triangle Signal (continuous)", fontweight="bold") 
 plt.plot(x,y,marker="o") 
 plt.grid(True) 
 plt.show() 
triangle_signal_contn() 

Triangular Wave(discrete):
import numpy as np
import matplotlib.pyplot as plt
def triangle_signal_dis(): 
 x = range(0, 10)
 y = []
 for i in x:
 y.append(3*abs(((i-1) % 4) - 2) - 3) 
 plt.figure(figsize=(4,4))
 plt.xlabel("t in secs", fontweight="bold") 
 plt.ylabel("T(t)", fontweight="bold")
 plt.title("Triangle Signal (Discrete)", fontweight="bold") 
 plt.stem(x,y)
 plt.grid(True)
 plt.show() 
triangle_signal_dis()

Rectangular Wave(continuous): 
import numpy as np 
import matplotlib.pyplot as plt 
def rectangular_wave(t, period, duty_cycle): 
 frequency = 1 / period 
 phase_shift = duty_cycle * period 
 return np.where((t % period) < phase_shift, 1, 0) 
t = np.linspace(0, 2, 1000) 
period = 1 
duty_cycle = 0.5 
rectangular_wave_signal = rectangular_wave(t, period, duty_cycle) 
plt.plot(t, rectangular_wave_signal) 
plt.title('Continuous Rectangular Wave') 
plt.xlabel('Time (s)') 
plt.ylabel('Amplitude') 
plt.grid(True) 
plt.show() 

Rectangular Wave(discrete): 
import numpy as np
import matplotlib.pyplot as plt
def rectangular_wave(t, period, duty_cycle):
 frequency = 1 / period
 phase_shift = duty_cycle * period
 return np.where((t % period) < phase_shift, 1, 0)
t = np.linspace(0, 2, 10) 
period = 1
duty_cycle = 0.5 
rectangular_wave_signal = rectangular_wave(t, period, duty_cycle)
plt.stem(t, rectangular_wave_signal)
plt.title('Discrete Rectangular Wave')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.grid(True)
plt.show()

Increasing Exponential(continuous): 
import math 
import matplotlib.pyplot as plt 
def exp_signal_contn(): 
 x = [] 
 y = [] 
 for i in range(-10, 10): 
 x.append(i) 
 y.append(math.exp(i)) 
 plt.figure(figsize=(4,4)) 
 plt.xlabel("t in secs", fontweight="bold") 
 plt.ylabel("E(t)", fontweight="bold") 
 plt.title("Exponential Signal Continuous", fontweight="bold") 
 plt.plot(x,y, marker="o") 
 plt.show() 
exp_signal_contn() 

Increasing Exponential (discrete): 
import math
import matplotlib.pyplot as plt
def exp_signal_dis(): 
 x = []
 y = []
 for i in range(-10, 10): 
 x.append(i) 
 y.append(math.exp(i))
 plt.figure(figsize=(4,4))
 plt.xlabel("t in secs", fontweight="bold") 
 plt.ylabel("E(t)", fontweight="bold")
 plt.title("Exponential Signal Discrete", fontweight="bold") 
 plt.stem(x,y)
 plt.show() 
exp_signal_dis()
