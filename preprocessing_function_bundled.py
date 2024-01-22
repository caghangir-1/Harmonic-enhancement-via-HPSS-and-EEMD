from scipy signal import iirnotch, butter, iirfilter, resample, filtfilt, find_peaks, savgol_filter
from scipy import signal
import emd

def nextpow2(p):     
    n = 2
    power=1
    while p > n:  
        n *= 2
        power += 1
    return power

def envelopeCreator(timeSignal, degree=3, intervalLength=51, hilbert_transform=False):
    
    if(hilbert_transform == True):
        timeSignal = np.abs(hilbert(timeSignal))
        
    amplitude_envelopeFiltered = savgol_filter(timeSignal, intervalLength, degree)
    return amplitude_envelopeFiltered  

def BARE(oscillation_waveform, height=5):

    # ===== Initiation =====
    peaks, properties = find_peaks(oscillation_waveform, distance=600, height=height)
    peaks_neg, properties = find_peaks(oscillation_waveform * -1, distance=600, height=height)

    q3 = np.percentile(oscillation_waveform, 75)
    q1 = np.percentile(oscillation_waveform, 25)
    
    # ========= 10 based log + infinity removal ========
    raw_log = np.log10(np.abs(oscillation_waveform))
    raw_log_inf_indices = np.where(np.isinf(raw_log))[0]
    for i in range(len(raw_log_inf_indices)):
        raw_log[raw_log_inf_indices[i]] = (raw_log[raw_log_inf_indices[i] - 1] + raw_log[raw_log_inf_indices[i] + 1]) / 2
    # ========= 10 based log + infinity removal ========
    
    raw_log_savgol = envelopeCreator(timeSignal=raw_log, degree=5, intervalLength=1001)
    # ===== Initiation =====
    
    # ====== Recursive peak removal algorithm =======
    for i in range(len(peaks)):
        
        # ===== Determine blinking period ======
        begin_dist, end_dist = 0, 0
        while(oscillation_waveform[peaks[i] - begin_dist] > q3):
            begin_dist += 1

        while(oscillation_waveform[peaks[i] + end_dist] > q3 and peaks[i] + end_dist < len(oscillation_waveform)-2):
            end_dist += 1
            
        begin_index, end_index = peaks[i] - begin_dist, peaks[i] + end_dist
        # ===== Determine blinking period ======
        
        # ===== Exchange data with log + sav-gol ======
        oscillation_waveform[begin_index : end_index] = raw_log_savgol[begin_index : end_index]
        # ===== Exchange data with log + sav-gol ======
        
    for i in range(len(peaks_neg)):
        
        # ===== Determine blinking period ======
        begin_dist, end_dist = 0, 0
        while(oscillation_waveform[peaks_neg[i] - begin_dist] < q1):
            begin_dist += 1
        while(oscillation_waveform[peaks_neg[i] + end_dist] < q1 and peaks_neg[i] + end_dist < len(oscillation_waveform)-2):
            end_dist += 1
            
        begin_index, end_index = peaks_neg[i] - begin_dist, peaks_neg[i] + end_dist
        # ===== Determine blinking period ======
        
        # ===== Exchange data with log + sav-gol ======
        oscillation_waveform[begin_index : end_index] = raw_log_savgol[begin_index : end_index] * -1
        # ===== Exchange data with log + sav-gol ======
        
    # ====== Recursive peak removal algorithm =======
    
    return oscillation_waveform

def harmonic_enhancement_preprocessing_pipeline(oscillation_waveform, Fs, normalization=True, resample=None, filter_type='butter', l_freq=2, h_freq=200, ifHPSS='harmonic', ifEEMD=True, remove_imf=None, if_peak_removal=True):
    
    # ======== Initial preprocessing =========    
    if(resample is not None):
        oscillation_waveform = scipy.resample(oscillation_waveform, resample)
    
    # ===== Notch filter =====
    f0 = 50 # Hz
    Q = 30.0  # Quality factor
    w0 = f0/(Fs/2)
    b, a = iirnotch(w0, Q)
    oscillation_waveform = filtfilt(b, a, oscillation_waveform)
    # ===== Notch filter ====
    
    # ====== Band-pass filter ======
    nyq = 0.5 * Fs
    low = l_freq / nyq
    high = h_freq / nyq
    if(filter_type == 'butter'):
        b, a = butter(order, [low, high], btype='band', analog=False)
    elif(filter_type == 'iir'):
        b, a = iirfilter(order, [low, high], btype='bandpass', analog=False, ftype='butter')
        
    oscillation_waveform = filtfilt(b, a, oscillation_waveform)
    # ====== Band-pass filter ======
    
    # ======== Initial preprocessing =========
    
    # ========= Robust z-score normalization =========
    if(normalization==True):
        MAD = np.median(np.abs(oscillation_waveform - np.median(oscillation_waveform)))
        oscillation_waveform = 0.6745 * (oscillation_waveform - np.median(oscillation_waveform)) / MAD
    # ========= Robust z-score normalization =========
    
    n_fft = 2 ** (nextpow2(Fs) + 2)
    
    # ======== Harmonic filtering ========
    D = librosa.stft(oscillation_waveform, n_fft=n_fft)
    
    if(ifHPSS == 'harmonic'):
        D_harmonic, D_percussive = librosa.decompose.hpss(D)
        oscillation_waveform = librosa.istft(D_harmonic, length=len(oscillation_waveform))        
    elif(ifHPSS == 'percussive'):
        D_harmonic, D_percussive = librosa.decompose.hpss(D)
        oscillation_waveform = librosa.istft(D_percussive, length=len(oscillation_waveform))
    # ======== Harmonic filtering ========
        
    # ======== EEMD =========
    if(ifEEMD == True):
        imf_opts = {'sd_thresh': 0.05}
        imf = emd.sift.ensemble_sift(oscillation_waveform, max_imfs=5, nensembles=96, nprocesses=6, ensemble_noise=1, imf_opts=imf_opts)
        
        if(remove_imf == '1st'):
            oscillation_waveform = np.sum(imf[:, 1:], axis=1)
        elif(remove_imf == 'last'):
            oscillation_waveform = np.sum(imf[:, 0:-1], axis=1)
        else:
            oscillation_waveform = np.sum(imf, axis=1)
    # ======== EEMD =========
    
    # ========= Recursive peak removal ===========
    if(if_peak_removal == True):
        oscillation_waveform = BARE(oscillation_waveform, height=5) #play with the height number based on your amplitude scale
    # ========= Recursive peak removal ===========
    
    return oscillation_waveform
