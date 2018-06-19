clc; clear; close all;

wav_file = dir('/home/rhythm/Desktop/m/mlsp/final_code/feature_extraction/data/*.wav');

[y, Fs] = audioread(['/home/rhythm/Desktop/m/mlsp/final_code/feature_extraction/data/' wav_file.name]);
frameSize = ceil(20e-3*Fs);
frameShift = ceil(10e-3*Fs);

% For MFCC features
mfcc=melcepst(y,Fs,'0dD',12,floor(3*log(Fs)),frameSize,frameShift);
%%%%%%%%%%%%%%%%%%%%%%% Mean normalization %%%%%%%%%%%%%%%%%%%%%%%
mfcc = mfcc - repmat(mean(mfcc), size(mfcc,1), 1);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
[~,nm,~] = fileparts(wav_file.name);
outPath = ['/home/rhythm/Desktop/m/mlsp/final_code/feature_extraction/mfcc/'];
if ~isdir(outPath)
    mkdir(outPath);
end
dlmwrite([outPath,nm,'.mfcc'],mfcc,' ');
    

        
