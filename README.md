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

Decreasing Exponential(continuous):
import math
import matplotlib.pyplot as plt
def exp_signal_contn(): 
 x = []
 y = []
 for i in range(-10, 10): 
  x.append(i) 
  y.append(math.exp(-i))
 plt.figure(figsize=(4,4))
 plt.xlabel("t in secs", fontweight="bold") 
 plt.ylabel("E(t)", fontweight="bold")
 plt.title("Exponential Signal Continuous", fontweight="bold") 
 plt.plot(x,y, marker="o")
 plt.show() 
exp_signal_contn() 

Decreasing Exponential (discrete): 
import math 
import matplotlib.pyplot as plt 
def exp_signal_dis(): 
 x = [] 
 y = [] 
 for i in range(-10, 10): 
  x.append(i) 
  y.append(math.exp(-i)) 
 plt.figure(figsize=(4,4)) 
 plt.xlabel("t in secs", fontweight="bold") 
 plt.ylabel("E(t)", fontweight="bold") 
 plt.title("Exponential Signal Discrete", fontweight="bold") 
 plt.stem(x,y) 
 plt.show() 
exp_signal_dis() 

Sawtooth(continuous): 
import math 
import matplotlib.pyplot as plt 
def sawtooth_contn(): 
 x = range(0, 20) 
 y = [] 
 for i in x: 
  y.append(2*((i/4) - math.floor((i/4) + (1/2)))) 
 plt.figure(figsize=(4,4)) 
 plt.xlabel("t in secs", fontweight="bold") 
 plt.ylabel("s(t) (amplitude)", fontweight="bold") 
 plt.title("Saw Tooth Signal (Continuous)", fontweight="bold") 
 plt.plot(x,y,marker="o") 
 plt.show() 
sawtooth_contn()

Sawtooth(discrete): 
import math
import matplotlib.pyplot as plt
def sawtooth_dis(): 
 x = range(0, 20)
 y = []
 for i in x:
  y.append(2*((i/4) - math.floor((i/4) + (1/2)))) 
 plt.figure(figsize=(4,4))
 plt.xlabel("t in secs", fontweight="bold")
 plt.ylabel("s(t) (amplitude)", fontweight="bold")
 plt.title("Saw Tooth Signal (discrete)", fontweight="bold") 
 plt.stem(x,y)
 plt.show()
sawtooth_dis()

Unit impusle(continuous): 
import numpy as np 
import matplotlib.pyplot as plt 
def impulse_signal_contn(): 
 x = np.arange(-1, 1.1, 0.1) 
 y = [] 
 for i in x: 
  if i <= -0.5 or i >= 0.5: 
   y.append(0) 
  else: 
   y.append(1) 
 plt.figure(figsize=(4,4)) 
 plt.xlabel("t in secs", fontweight="bold") 
 plt.ylabel("pulse(t)", fontweight="bold") 
 plt.title("Impulse Signal (continuous)", fontweight= "bold") 
 plt.plot(x,y, marker="o") 
 plt.show() 
impulse_signal_contn() 

Unit impusle(discrete): 
import numpy as np
import matplotlib.pyplot as plt
def impulse_signal_dis():
 x = np.arange(-1, 1.1, 0.1) 
 y = []
 for i in x:
  if i <= -0.5 or i >= 0.5: 
   y.append(0)
  else:
   y.append(1)
 plt.figure(figsize=(4,4))
 plt.xlabel("t in secs", fontweight="bold") 
 plt.ylabel("pulse(t)", fontweight="bold")
 plt.title("Impulse Signal (discrete)", fontweight= "bold") 
 plt.stem(x,y)
 plt.show() 
impulse_signal_dis()

Unit step(continuous): 
import numpy as np 
import matplotlib.pyplot as plt 
def unit_step_contn(): 
 x = [] 
 y = [] 
 for i in range(-10, 11): 
  if i >= -0.1: 
   x.append(i) 
   y.append(1) 
  elif i <= -0.1: 
   x.append(i) 
   y.append(0) 
 plt.figure(figsize=(4,4)) 
 plt.xlabel("t in secs", fontweight="bold") 
 plt.ylabel("u(t)", fontweight="bold") 
 plt.title("Unit Step signal (Continuous)", fontweight="bold") 
 plt.plot(x,y,marker="o") 
 plt.show() 
unit_step_contn() 

Unit step(discrete): 
import numpy as np
import matplotlib.pyplot as plt
def unit_step_dis(): 
 x = []
 y = []
 for i in range(-10, 11): 
  if i >= -0.1:
   x.append(i) 
   y.append(1) 
  elif i <= -0.1: 
   x.append(i) 
   y.append(0)
 plt.figure(figsize=(4,4))
 plt.xlabel("t in secs", fontweight="bold") 
 plt.ylabel("u(t)", fontweight="bold")
 plt.title("Unit Step signal (Discrete)", fontweight="bold") 
 plt.stem(x,y)
 plt.show() 
unit_step_dis()

Unit ramp(continuous): 
import numpy as np 
import matplotlib.pyplot as plt 
def unit_ramp_contn(): 
 x = [] 
 y = [] 
 for i in range(-10, 11): 
  if i >= 0: 
   x.append(i) 
   y.append(i) 
  elif i < 0: 
   x.append(i) 
   y.append(0) 
 plt.figure(figsize=(4,4)) 
 plt.xlabel("t in secs", fontweight="bold") 
 plt.ylabel("r(t)", fontweight="bold") 
 plt.title("Unit Ramp Signal Continuous", fontweight="bold") 
 plt.plot(x,y,marker="o") 
 plt.show() 
unit_ramp_contn() 

Unit ramp(discrete): 
import numpy as np
import matplotlib.pyplot as plt 
def unit_ramp_dis(): 
 x = []
 y = []
 for i in range(-10, 11): 
  if i >= 0:
   x.append(i) 
   y.append(i)
  elif i < 0: 
   x.append(i) 
   y.append(0)
 plt.figure(figsize=(4,4))
 plt.xlabel("t in secs", fontweight="bold") 
 plt.ylabel("r(t)", fontweight="bold")
 plt.title("Unit Ramp Signal Discrete", fontweight="bold") 
 plt.stem(x,y)
 plt.show() 
unit_ramp_dis()

Gaussian(continuous): 
import numpy as np 
import matplotlib.pyplot as plt 
t = np.linspace(-5, 5, 1000) 
mu = 0 
sigma = 1 
gaussian_wave_signal = np.exp(-0.5 * ((t - mu) / sigma)**2) 
plt.plot(t, gaussian_wave_signal) 
plt.title('Gaussian Wave Signal') 
plt.xlabel('Time (s)') 
plt.ylabel('Amplitude') 
plt.grid(True) 
plt.show()

Gaussian(discrete): 
import numpy as np
import matplotlib.pyplot as plt
t = np.linspace(-5, 5, 30)
mu = 0 
sigma = 1 
gaussian_wave_signal = np.exp(-0.5 * ((t - mu) / sigma)**2)
plt.stem(t, gaussian_wave_signal)
plt.title('Gaussian Wave Signal')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.grid(True)
plt.show()

Addition(continuous): 
import numpy as np 
import matplotlib.pyplot as plt 
signal1 = np.array([1, 2, 5, 7, 3]) # First signal 
signal2 = np.array([2, 4, 5, 2, 2]) # Second signal 
sum_signal = signal1 + signal2 
t = np.linspace(0, len(signal1) - 1, len(signal1)) # Assuming unit spacing between samples 
plt.figure(figsize=(8, 6)) 
plt.plot(t, signal1, marker='o', label='Signal 1') 
plt.plot(t, signal2, marker='o', label='Signal 2') 
plt.plot(t, sum_signal, marker='o', label='Sum of Signals') 
plt.title('Signals') 
plt.xlabel('Time (Index)') 
plt.ylabel('Amplitude') 
plt.grid(True) 
plt.show()

Addition(discerete):
import numpy as np
import matplotlib.pyplot as plt
signal1 = np.array([1, 2, 5, 7, 3]) # First signal
signal2 = np.array([2, 4, 5, 2, 2]) # Second signal
sum_signal = signal1 + signal2
t = np.linspace(0, len(signal1) - 1, len(signal1)) # Assuming unit spacing between samples
plt.figure(figsize=(6, 4))
plt.subplot(3, 1, 1)
plt.stem(t, signal1, label='Signal 1')
plt.title('Signals')
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.subplot(3, 1, 2) 
plt.stem(t, signal2, label='Signal 2') 
plt.xlabel('Time') 
plt.ylabel('Amplitude') 
plt.subplot(3, 1, 3) 
plt.stem(t, sum_signal, label='Sum of Signals') 
plt.xlabel('Time') 
plt.ylabel('Amplitude') 
plt.tight_layout() 
plt.show()

Subtraction(continuous): 
import numpy as np 
import matplotlib.pyplot as plt 
signal1 = np.array([1, 2, 5, 7, 3]) # First signal 
signal2 = np.array([2, 4, 5, 2, 2]) # Second signal 
sum_signal = signal1 - signal2 
t = np.linspace(0, len(signal1) - 1, len(signal1)) # Assuming unit spacing between samples 
plt.figure(figsize=(4, 3)) 
plt.plot(t, signal1, marker='o', label='Signal 1') 
plt.plot(t, signal2, marker='o', label='Signal 2') 
plt.plot(t, sum_signal, marker='o', label='Sum of Signals') 
plt.title('Signals') 
plt.xlabel('Time (Index)') 
plt.ylabel('Amplitude') 
plt.grid(True) 
plt.show()

Subtraction(discrete)
import numpy as np
import matplotlib.pyplot as plt
signal1 = np.array([1, 2, 5, 7, 3]) # First signal
signal2 = np.array([2, 4, 5, 2, 2]) # Second signal
sum_signal = signal1 - signal2
t = np.linspace(0, len(signal1) - 1, len(signal1)) # Assuming unit spacing between samples
plt.figure(figsize=(7, 5))
plt.subplot(3, 1, 1)
plt.stem(t, signal1, label='Signal 1')
plt.title('Signals')
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.subplot(3, 1, 2) 
plt.stem(t, signal2, label='Signal 2') 
plt.xlabel('Time') 
plt.ylabel('Amplitude') 
plt.subplot(3, 1, 3) 
plt.stem(t, sum_signal, label='Sum of Signals') 
plt.xlabel('Time') 
plt.ylabel('Amplitude') 
plt.tight_layout() 
plt.show() 

Multiplication(continuous): 
import numpy as np 
import matplotlib.pyplot as plt 
signal1 = np.array([1, 2, 5, 7, 3]) # First signal 
signal2 = np.array([2, 4, 5, 2, 2]) # Second signal 
sum_signal = signal1 * signal2 
t = np.linspace(0, len(signal1) - 1, len(signal1)) # Assuming unit spacing between samples 
plt.figure(figsize=(4, 3)) 
plt.plot(t, signal1, marker='o', label='Signal 1') 
plt.plot(t, signal2, marker='o', label='Signal 2') 
plt.plot(t, sum_signal, marker='o', label='Sum of Signals') 
plt.title('Signals') 
plt.xlabel('Time (Index)') 
plt.ylabel('Amplitude') 
plt.grid(True) 
plt.show()

Multiplication(discrete):
import numpy as np
import matplotlib.pyplot as plt
signal1 = np.array([1, 2, 5, 7, 3]) # First signal
signal2 = np.array([2, 4, 5, 2, 2]) # Second signal
sum_signal = signal1 * signal2
t = np.linspace(0, len(signal1) - 1, len(signal1)) # Assuming unit spacing between samples
plt.figure(figsize=(7, 5))
plt.subplot(3, 1, 1)
plt.stem(t, signal1, label='Signal 1')
plt.title('Signals')
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.subplot(3, 1, 2) 
plt.stem(t, signal2, label='Signal 2') 
plt.xlabel('Time') 
plt.ylabel('Amplitude') 
plt.subplot(3, 1, 3) 
plt.stem(t, sum_signal, label='Sum of Signals') 
plt.xlabel('Time') 
plt.ylabel('Amplitude') 
plt.tight_layout() 
plt.show() 

Sampling
import numpy as np 
import matplotlib.pyplot as plt 
# Clear console 
plt.clf() 
T = 0.04 # Time period of 50 Hz signal 
t = np.arange(0, 0.02, 0.0005) 
f = 1 / T 
n1 = np.arange(0, 41) 
# Continuous signal 
xa_t = np.sin(2 * np.pi * 2 * t / T) 
plt.subplot(2, 2, 1) 
plt.plot(200 * t, xa_t) 
plt.title('Verification of sampling theorem') 
plt.xlabel('t') 
plt.ylabel('x(t)') 
ts1 = 0.002 # >niq rate 
ts2 = 0.01 # =niq rate 
ts3 = 0.1 # <niq rate 
n = np.arange(0, 21) 
x_ts1 = 2 * np.sin(2 * np.pi * n * ts1 / T) 
plt.subplot(2, 2, 2) 
plt.stem(n, x_ts1) 
plt.title('greater than Nq') 
plt.xlabel('n') 
plt.ylabel('x(n)') 
n = np.arange(0, 5) 
x_ts2 = 2 * np.sin(2 * np.pi * n * ts2 / T) 
plt.subplot(2, 2, 3) 
plt.stem(n, x_ts2) 
plt.title('Equal to Nq') 
plt.xlabel('n') 
plt.ylabel('x(n)') 
n = np.arange(0, 11) 
x_ts3 = 2 * np.sin(2 * np.pi * n * ts3 / T) 
plt.subplot(2, 2, 4) 
plt.stem(n, x_ts3) 
plt.title('less than Nq') 
plt.xlabel('n') 
plt.ylabel('x(n)') 
plt.tight_layout() 
plt.show()

Conv Command 
import numpy as np 
import matplotlib.pyplot as plt 
# Clear console 
plt.clf() 
# Input sequences 
x1 = np.array(input('Enter the first sequence: ').split(), dtype=float) 
x2 = np.array(input('Enter the second sequence: ').split(), dtype=float) 
# Plot the first sequence 
plt.subplot(3, 1, 1) 
plt.stem(x1) 
plt.ylabel('Amplitude') 
plt.title('Plot of the first sequence') 
# Plot the second sequence 
plt.subplot(3, 1, 2) 
plt.stem(x2) 
plt.ylabel('Amplitude') 
plt.title('Plot of the second sequence') 
# Linear convolution 
f = np.convolve(x1, x2, mode='full') 
print('Output of linear convolution is:') 
print(f) 
# Plot the linear convolution result 
plt.subplot(3, 1, 3) 
plt.stem(f) 
plt.xlabel('Time index n') 
plt.ylabel('Amplitude f') 
plt.title('Linear convolution of sequences') 
plt.tight_layout() 
plt.show() 

Linear convultion using DFT and IDFT 
import numpy as np 
import matplotlib.pyplot as plt 
# Clear console 
plt.clf() 
# Input sequences 
x1 = np.array(input('Enter the first sequence: ').split(), dtype=float) 
x2 = np.array(input('Enter the second sequence: ').split(), dtype=float) 
n = int(input('Enter the number of points of the DFT: ')) 
# Plot the first sequence 
plt.subplot(3, 1, 1) 
plt.stem(x1, markerfmt='C0o', linefmt='C0-') 
plt.title('Plot of the first sequence') 
# Plot the second sequence 
plt.subplot(3, 1, 2) 
plt.stem(x2, markerfmt='C1o', linefmt='C1-') 
plt.title('Plot of the second sequence') 
# Lengths of sequences 
n1 = len(x1) 
n2 = len(x2) 
# Linear convolution 
m = n1 + n2 - 1 # Length of linear convolution 
x = np.concatenate((x1, np.zeros(n2 - 1))) # Padding of zeros to make it of length m 
y = np.concatenate((x2, np.zeros(n1 - 1))) 
x_fft = np.fft.fft(x, m) 
y_fft = np.fft.fft(y, m) 
dft_xy = x_fft * y_fft 
y = np.fft.ifft(dft_xy, m) 
print('The circular convolution result is:') 
print(y.real) 
# Plot the circularly convoluted sequence 
plt.subplot(3, 1, 3) 
plt.stem(y.real, markerfmt='C2o', linefmt='C2-') 
plt.title('Plot of circularly convoluted sequence') 
plt.tight_layout() 
plt.show() 

Circular convolution 
import numpy as np 
import matplotlib.pyplot as plt 
# Clear console 
plt.clf() 
# Input sequences 
x1 = np.array(input('Enter the first sequence: ').split(), dtype=float) 
x2 = np.array(input('Enter the second sequence: ').split(), dtype=float) 
n = max(len(x1), len(x2)) 
# Plot the first sequence 
plt.subplot(3, 1, 1) 
plt.stem(x1, markerfmt='C0o', linefmt='C0-') 
plt.title('Plot of the first sequence') 
# Plot the second sequence 
plt.subplot(3, 1, 2) 
plt.stem(x2, markerfmt='C1o', linefmt='C1-') 
plt.title('Plot of the second sequence') 
# Compute circular convolution 
y1 = np.fft.fft(x1, n) 
y2 = np.fft.fft(x2, n) 
y3 = y1 * y2 
y = np.fft.ifft(y3, n) 
print('The circular convolution result is:') 
print(y.real) 
# Plot the circularly convoluted sequence 
plt.subplot(3, 1, 3)
plt.stem(y.real, markerfmt='C2o', linefmt='C2-')
plt.title('Plot of circularly convoluted sequence')
plt.tight_layout()
plt.show()

DIF Algorithm 
import numpy as np 
import math 
xn = [1,2,3,4,4,3,2,1] 
w20 = w40 = w80 = 1 
w41 = w82 = -1j 
w81 = (1 - 1j) / (math.sqrt(2)) 
w83 = (-1 - 1j) / (math.sqrt(2)) 
# STAGE 1 - RADIX 8 POINT 
y3n = [0] * 8 
y3n[0] = xn[0] + xn[4] 
y3n[1] = xn[1] + xn[5] 
y3n[2] = xn[2] + xn[6] 
y3n[3] = xn[3] + xn[7] 
y3n[4] = (xn[0] - xn[4]) * w80 
y3n[5] = (xn[1] - xn[5]) * w81 
y3n[6] = (xn[2] - xn[6]) * w82 
y3n[7] = (xn[3] - xn[7]) * w83 
print("Stage 1 - 8 point solution:") 
print(y3n) 
# STAGE 2 - 4 POINT 
y2n = [0] * 8 
y2n[0] = y3n[0] + y3n[2] 
y2n[1] = y3n[1] + y3n[3] 
y2n[2] = (y3n[0] - y3n[2]) * w40 
y2n[3] = (y3n[1] - y3n[3]) * w41 
y2n[4] = y3n[4] + y3n[6] 
y2n[5] = y3n[5] + y3n[7] 
y2n[6] = (y3n[4] - y3n[6]) * w40 
y2n[7] = (y3n[5] - y3n[7]) * w41 
print("\nStage 2 - 4 point solution:") 
print(y2n) 
# Stage 3 - 2 POINT 
y1n = [0] * 8 
y1n[0] = y2n[0] + y2n[1] 
y1n[1] = y2n[4] + y2n[5] 
y1n[2] = y2n[2] + y2n[3] 
y1n[3] = y2n[6] + y2n[7] 
y1n[4] = (y2n[4] - y2n[5]) * w20 
y1n[5] = (y2n[2] - y2n[3]) * w20 
y1n[6] = (y2n[6] - y2n[7]) * w20 
print("\nStage 3 - 2 point solution:") 
print("Solution obtained using computation of DIF FFT") 
print(y1n) 
print("\nExact solution using FFT function") 
print(np.fft.fft(xn)) 

DIT Algorithm 
import numpy as np 
import math 
xn = [1,2,3,4,4,3,2,1] 
w2o=w4o=w8o=1 
w41=w82=-1j 
w81=(1-1j)/(math.sqrt(2)) 
w83=(-1-1j)/(math.sqrt(2)) 
y1n = [] 
for i in range(0,4): 
 y1n.append(xn[i]+xn[i+4]*w2o) 
 y1n.append(xn[i]-xn[i+4]*w2o) 
print("The answer after 1st stage obtained by two point butterfly structure:") 
print(y1n) 
y2n=[] 
y3n=[] 
for i in range(0,8): 
 y2n.append(0) 
 y3n.append(0) 
k=0 
for i in range(0,2): 
 if k%2==0: 
  y2n[k]=y1n[i]+y1n[i+2]*w4o 
  y2n[k+4]=y1n[i]+y1n[i+2]*w4o 
  y2n[k+2]=y1n[i]-y1n[i+2]*w4o 
  y2n[k+6]=y1n[i]-y1n[i+2]*w4o 
 else: 
  y2n[k]=y1n[i]+y1n[i+2]*w41 
  y2n[k+4]=y1n[i]+y1n[i+2]*w41 
  y2n[k+2]=y1n[i]-y1n[i+2]*w41 
  y2n[k+6]=y1n[i]-y1n[i+2]*w41 
 k=k+1 
 
print("The answer after first stage obtained by 4 point butterfly structure:") 
print(y2n) 
y3n[0]=y2n[0]+y2n[4]*w8o 
y3n[1]=y2n[1]+y2n[5]*w81 
y3n[2]=y2n[2]+y2n[6]*w82 
y3n[3]=y2n[3]+y2n[7]*w83 
y3n[4]=y2n[0]-y2n[4]*w8o 
y3n[5]=y2n[1]-y2n[5]*w81 
y3n[6]=y2n[2]-y2n[6]*w82 
y3n[7]=y2n[3]-y2n[7]*w83 
print("Final answer after 8 point structure:") 
print(y3n) 
print("The exact answer found using python fft function") 
print(np.fft.fft(xn))
