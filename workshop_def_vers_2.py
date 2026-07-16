import numpy as np
import logging
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s', handlers=[logging.StreamHandler(sys.stdout)])

# =================================================================
# 1. RESERVOIR COMPARTIMENTALE CON LEGGE DI DALE
# =================================================================
class CompartmentalReservoirDale:
    def __init__(self, n_neurons=32, input_dim=1, frac_inh=0.2, dt=1.0):
        self.N = n_neurons
        self.dt = dt
        self.tau_m    = 30.0
        self.v_rest   = -70.0
        self.v_thresh = -55.0
        self.v_reset  = -75.0
        
        # Suddivisione popolazioni: primi N_inh indici = Inibitori, resto = Eccitatori
        self.N_inh = int(self.N * frac_inh)
        self.N_exc = self.N - self.N_inh
        
        # SFA (Memoria a lungo termine)
        self.tau_adapt = 3500.0  
        self.beta      = 2.0    
        self.tau_s     = 10.0
        
        # Inizializzazione pesi sparsi assoluti (strettamente positivi)
        pesi_assoluti = np.abs(np.random.randn(self.N, self.N) * 2.0)
        maschera_sparsa = np.random.rand(self.N, self.N) < 0.30
        self.W_res = pesi_assoluti * maschera_sparsa
        
        # APPLICAZIONE LEGGE DI DALE BILANCIATA
        # Moltiplichiamo per 0.4 per evitare di sopprimere totalmente l'attività della rete
        self.W_res[:, :self.N_inh] = -self.W_res[:, :self.N_inh] * 0.4
        np.fill_diagonal(self.W_res, 0.0)
        
        # Pesi di input: l'input stimola i neuroni
        self.W_in = np.random.uniform(15.0, 45.0, (self.N, input_dim))
        self.reset_state()

    def reset_state(self):
        self.v_m     = np.full(self.N, self.v_rest)
        self.i_syn   = np.zeros(self.N)
        self.v_adapt = np.zeros(self.N) 
        self.spikes  = np.zeros(self.N, dtype=bool)

    def step(self, input_signal):
        # SFA attiva solo sui neuroni eccitatori (semplificazione biologica comune)
        self.v_adapt[self.N_inh:] += (-self.v_adapt[self.N_inh:]) / self.tau_adapt * self.dt
        v_thresh_eff = np.full(self.N, self.v_thresh)
        v_thresh_eff[self.N_inh:] += self.v_adapt[self.N_inh:]
        
        # L'input agisce come iniezione di corrente
        external_jump = self.W_in @ input_signal
        
        # Corrente sinaptica integrata nel tempo
        di_syn = (-self.i_syn) / self.tau_s * self.dt
        self.i_syn += di_syn + self.spikes * 40.0 + external_jump
        
        recurrent_input = self.W_res @ self.i_syn
        
        dv_m = (-(self.v_m - self.v_rest) + recurrent_input + self.i_syn) / self.tau_m
        self.v_m += dv_m * self.dt
        
        self.spikes = self.v_m >= v_thresh_eff
        self.v_m[self.spikes] = self.v_reset
        
        # Incremento SFA solo per gli eccitatori
        self.v_adapt[self.N_inh:][self.spikes[self.N_inh:]] += self.beta
        
        return self.spikes.copy()


# =================================================================
# 2. SPIKING READOUT (Gated via VIP-SST)
# =================================================================
class SpikingReadout:
    def __init__(self, n_neurons_in=32, output_dim=1, dt=1.0):
        self.N_in   = n_neurons_in
        self.N_out  = output_dim
        self.dt     = dt
        self.tau_m  = 20.0
        self.tau_s  = 5.0
        self.v_rest   = -70.0
        self.v_thresh = -55.0
        self.v_reset  = -75.0
        
        self.W_out    = np.random.rand(self.N_out, self.N_in) * 5.0
        # AUMENTATO DA 100.0 A 200.0 PER FILTRARE LE OSCILLAZIONI E/I
        self.tau_trace = 50.0 
        self.reset_state()

    def reset_state(self):
        self.v_m     = np.full(self.N_out, self.v_rest) 
        self.i_syn   = np.zeros(self.N_out)
        self.r_trace = np.zeros(self.N_in)
        self.o_trace = np.zeros(self.N_out)
        self.t_trace = np.zeros(self.N_out)
    
    def step(self, reservoir_spikes, target_spikes=None, lr=0.1, train=False):
        self.r_trace += (-self.r_trace + reservoir_spikes.astype(float)) / self.tau_trace * self.dt
        input_current = self.W_out @ self.r_trace 
        
        # Dis-Inibizione VIP -> SST
        sst_inhibition = 0.0 if train else 30.0  
        
        di_syn = (-self.i_syn + input_current - sst_inhibition) / self.tau_s * self.dt
        self.i_syn += di_syn
        
        dv_m = (-(self.v_m - self.v_rest) + self.i_syn) / self.tau_m * self.dt
        self.v_m += dv_m
        
        out_spikes = self.v_m >= self.v_thresh
        self.v_m[out_spikes] = self.v_reset
        
        self.o_trace += (-self.o_trace + out_spikes.astype(float)) / self.tau_trace * self.dt
        
        error_val = 0.0
        
        if target_spikes is not None:
            self.t_trace += (-self.t_trace + target_spikes.astype(float)) / self.tau_trace * self.dt
            
        if train:
            error = self.t_trace - self.o_trace
            self.W_out += lr * np.outer(error, self.r_trace)
            self.W_out = np.clip(self.W_out, 0.0, 15) 
            error_val = error[0]
            
        return out_spikes, error_val
    
# =================================================================
# 3. GENERAZIONE DATI
# =================================================================
def generate_poisson_spikes_numpy(frs, phase_dur_s=1.0, num_pres=6, dt=1.0):
    phase_steps = int(phase_dur_s * 1000.0 / dt)
    rates_hz = np.concatenate([[(f, 0.0)] * num_pres for f in frs]).flatten()
    total_steps = len(rates_hz) * phase_steps
    input_spikes = np.zeros(total_steps)
    rate_trajectory = np.zeros(total_steps)

    for i, rate in enumerate(rates_hz):
        start, end = i * phase_steps, (i + 1) * phase_steps
        prob = rate * (dt / 1000.0)
        input_spikes[start:end] = (np.random.rand(phase_steps) < prob).astype(float)
        rate_trajectory[start:end] = rate

    return input_spikes, rate_trajectory

def build_target_spikes(rate_trajectory, high_rate_hz=200, low_rate_hz=10, dt=1.0, max_learning_phases=100):
    Time = len(rate_trajectory)
    target_spikes   = np.zeros((Time, 1), dtype=bool)
    transition_mask = np.zeros(Time, dtype=bool)

    high_step = max(1, int((1000.0 / dt) / high_rate_hz))
    low_step  = max(1, int((1000.0 / dt) / low_rate_hz))
    transition_count = 0 

    for t in range(1, Time):
        if rate_trajectory[t] > 0 and rate_trajectory[t] != rate_trajectory[t-1]:
            transition_count += 1
            if transition_count <= max_learning_phases:
                end_trans = min(t + 500, Time)
                transition_mask[t:end_trans] = True

    for t in range(Time):
        if rate_trajectory[t] > 0:
            if transition_mask[t]:
                if t % high_step == 0: target_spikes[t] = True
            else:
                if t % low_step == 0: target_spikes[t] = True

    return target_spikes, transition_mask

# =================================================================
# 4. ESECUZIONE MAIN E PLOTTING
# =================================================================
def main():
    np.random.seed(0)

    teacher = CompartmentalReservoirDale(n_neurons=32, input_dim=1, frac_inh=0.2)
    readout = SpikingReadout(n_neurons_in=32, output_dim=1)

    frequenze_target = [40, 120, 80]
    input_history, rate_trajectory = generate_poisson_spikes_numpy(frs=frequenze_target, phase_dur_s=1.0, num_pres=6)
    Time = len(input_history)

    target_spikes_history, transition_mask = build_target_spikes(rate_trajectory, high_rate_hz=200, low_rate_hz=10)

    logging.info("Avvio Addestramento (Lazy Teacher con Legge di Dale)...")
    epochs = 250 # Aumentato da 150 a 200 per compensare il LR più basso
    for epoch in range(epochs):
        teacher.reset_state()
        readout.reset_state()
        epoch_error = 0.0
        learning_steps = 0

        for t in range(Time):
            spikes_res = teacher.step(np.array([input_history[t]]))
            is_teaching = transition_mask[t] 
            
            spikes_out, err = readout.step(
                spikes_res, 
                target_spikes=target_spikes_history[t], 
                lr=0.002, # Ridotto da 0.05 a 0.015 per evitare instabilità
                train=is_teaching 
            )
            
            if is_teaching:
                epoch_error += err ** 2
                learning_steps += 1

        if (epoch+1) % 20 == 0 and learning_steps > 0:
            logging.info(f"Epoca {epoch + 1:03d}/{epochs} | MSE (sulle transizioni): {(epoch_error / learning_steps):.6f}")

    # TEST FINALE E PLOT
    logging.info("Generazione Grafici in corso...")
    teacher.reset_state()
    readout.reset_state()

    plot_time, plot_res, plot_target, plot_readout, plot_error = [], [], [], [], []

    for t in range(Time):
        spikes_res = teacher.step(np.array([input_history[t]]))
        target_t   = target_spikes_history[t]
        
        spikes_out, _ = readout.step(spikes_res, target_spikes=target_t, train=False)

        plot_time.append(t)
        plot_res.append(np.where(spikes_res)[0])
        plot_target.append(target_t[0])
        plot_readout.append(spikes_out[0])
        plot_error.append(readout.t_trace[0] - readout.o_trace[0])

    readout_arr = np.array(plot_readout)
    kernel = np.ones(100) / 100
    readout_firing_rate_hz = np.convolve(readout_arr, kernel, mode='same') * 1000.0

    fig, axes = plt.subplots(5, 1, figsize=(16, 16), sharex=True)
    is_zero = (rate_trajectory == 0)
    time_arr = np.array(plot_time)

    for ax in axes:
        ax.fill_between(time_arr, 0, 1, where=is_zero, transform=ax.get_xaxis_transform(), color='gray', alpha=0.18, linewidth=0)
        ax.fill_between(time_arr, 0, 1, where=transition_mask, transform=ax.get_xaxis_transform(), color='#22c55e', alpha=0.45, linewidth=0)

    axes[0].set_title("1. Reservoir Activity (Red = Inhibitory, Black = Excitatory)")
    for t_idx, neurons in enumerate(plot_res):
        if len(neurons) > 0:
            # Separazione indici per il plotting
            inh_spikes = neurons[neurons < teacher.N_inh]
            exc_spikes = neurons[neurons >= teacher.N_inh]
            
            if len(inh_spikes) > 0:
                axes[0].scatter([t_idx] * len(inh_spikes), inh_spikes, color='red', s=2, marker='|', alpha=0.7)
            if len(exc_spikes) > 0:
                axes[0].scatter([t_idx] * len(exc_spikes), exc_spikes, color='black', s=2, marker='|')
                
    axes[0].axhline(teacher.N_inh - 0.5, color='gray', linestyle='--', lw=0.8) # Linea di separazione E/I
    axes[0].set_ylabel("ID Neurone")
    axes[0].set_ylim(-1, 33)

    axes[1].set_title("2. Input Frequency (Hz)")
    axes[1].plot(time_arr, rate_trajectory, color='orange', lw=1.5)
    axes[1].set_ylabel("Hz")

    axes[2].set_title("3. Raster Plot: Target (blue) vs Output (green)")
    target_idx  = np.where(np.array(plot_target))[0]
    readout_idx = np.where(readout_arr)[0]
    axes[2].vlines(target_idx, 0.6, 1.4, colors='blue', lw=0.2, label='Target')
    axes[2].vlines(readout_idx, -0.4, 0.4, colors='green', lw=0.2, label='Output')
    axes[2].set_ylim(-0.8, 1.8)
    axes[2].legend(loc='upper right')

    axes[3].set_title("4. Firing Rate Output (Decay and SST)")
    axes[3].plot(time_arr, readout_firing_rate_hz, color='green', lw=1.5)
    axes[3].axhline(200, color='blue', linestyle=':', lw=1, alpha=0.5, label='Target (200 Hz)')
    axes[3].axhline(10, color='black', linestyle=':', lw=1, alpha=0.5, label='Target (10 Hz)')
    axes[3].set_ylabel("Hz")
    axes[3].legend(loc='upper right')

    axes[4].set_title("5. Error")
    axes[4].plot(time_arr, np.array(plot_error), color='red', lw=1)
    axes[4].axhline(0, color='black', linestyle='--', lw=0.8)
    axes[4].set_xlabel("Tempo (ms)")

    plt.tight_layout()
    filename = "snn_dale_inibitorio.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    logging.info(f"[SUCCESSO] Esecuzione completata. Controlla il file '{filename}'")

if __name__ == "__main__":
    main()