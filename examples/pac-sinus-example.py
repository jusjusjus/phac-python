#!/usr/bin/env python

from os.path import dirname, join
from sys import path
path.insert(0, join(dirname(__file__), '..'))

import numpy as np
import matplotlib.pyplot as plt

from phac import phase_amplitude_coupling

pi2 = 2 * np.pi  # rad

sampling_rate = 128.0  # Hz
T = 3.0  # sec.
t = np.arange(0, T, 1 / sampling_rate)
f_slow = 2.0  # Hz
f_fast = 25.0  # Hz
A0 = 0.1  # a.u.
dA = 0.2  # a.u.
dphi = - 128 * np.pi / 180.0  # rad

slow = (1.5, 4.0)  # Hz
fast = (10., 30.0)  # Hz

phi_slow = pi2 * f_slow * t
x_slow = np.sin(phi_slow) + 0.2 * np.random.randn(t.size)

At = A0 + dA * (np.sin(phi_slow - dphi) + 1) ** 4 + 0.2 * np.random.randn(t.size)
x_fast = At * np.sin(pi2 * f_fast * t) + 0.2 * np.random.randn(t.size)
x = x_slow + x_fast

pac = phase_amplitude_coupling(x, sampling_rate, slow, fast)
mpc = pac.mean_phase_coherence
mpc_amp = np.abs(mpc)
mpc_phi = 180 * np.angle(mpc) / np.pi

plt.figure(figsize=(6, 12))
plt.subplot(211)
plt.title(f"MI={pac.modulation_index:.2g}")
plt.plot(t, x)

plt.subplot(212)
plt.title(f"MPC={mpc_amp:.2f} Amp, {mpc_phi:.0f} Grad")
angle = np.linspace(0, pi2, num=32)
plt.plot(np.cos(angle), np.sin(angle), 'k--', lw=2, alpha=0.5)
plt.plot(0, 0, 'k+', ms=12)
plt.plot(np.real(mpc), np.imag(mpc), 'ro')
plt.xticks([])
plt.yticks([])

plt.tight_layout()
plt.savefig("pac-sinus-example.png")
plt.show()
