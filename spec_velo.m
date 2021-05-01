% Espectro de velocidade
% Projeto WaveScatter
clear, clc, close all

pathname = '/home/hp/gdrive/coppe/lioc/wavescatter/output/paper_onr/20191121_v05/out/';

v = load([pathname, 'velo_paths_T100_050300_CAM1.csv']);

% media da velocidade
vm = mean(v,2);

X = vm - mean(vm);
Fs = 30;
NFFT = length(vm) / 4;
WINDOW = hann(NFFT);
NOVERLAP = round(length(WINDOW) / 2);

[pxx, freq] = pwelch(X,WINDOW,NOVERLAP,NFFT,Fs);

loglog(freq, pxx)
