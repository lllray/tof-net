% simulate phase images: 0,90,180,270
function [phase_imgs, corr_imgs] = GetPhaseImgs(hdrs,freqs,delays,isCmatReal)
    [nh,nw,nt] = size(hdrs);
    if ~exist('isCmatReal','var') || ~isCmatReal
        C = GetCorrelationMatrix(freqs,delays);
        nf = numel(freqs);
    else
        load processed_Cmat_0828; % load measured correlation matrix (kaust tintin w/ ext mod signal for phase sweep)
        nf = size(C,1)/2;
    end
    hdrs_t = reshape(hdrs,nh*nw,nt)';
    h = C*hdrs_t;
    h = reshape(h,2*nf,nh,nw);
    h0mat = h(1:nf,:,:); %cos
    h90mat = h(nf+1:end,:,:); %sin
    corr_imgs = h0mat + 1i*h90mat;
    phase_imgs = angle(corr_imgs);
    for fi = 1:nf
        tmp = squeeze(phase_imgs(fi,:,:)<0);
        phase_imgs(fi,tmp) = 2*pi + phase_imgs(fi,tmp);
    end
    corr_imgs = cat(1,h0mat,h90mat);
end


% %%%%%%%%%%%%%%%%Code From Phasor Imaging%%%%%%%%%%%%%%%%%%%%%
% % This function computes the phase-maps for a given frequency, and a set of
% % phase-shifted images, and the phase shift vector.
% 
% function [PhaseMap]  =  ComputePhaseMaps(IMat, shiftVec)
% 
% nr          = size(IMat, 1);
% nc          = size(IMat, 2);
% NumShifts   = numel(shiftVec);
% 
% %%%% Now solving the frequency equations
% A   = [ones(NumShifts,1)    cos(shiftVec)'   -sin(shiftVec)'];          %%% Assumes that the images are formed using cos(phi - delta)
% B   = reshape(IMat, [nr*nc NumShifts])';
% C   = A\B;
% clear A B
% 
% Amp                     = sqrt(C(2,:).^2 + C(3,:).^2);
% Phase                   = acos(C(2,:)./Amp);
% Phase((C(3,:)<0))       = 2*pi - Phase((C(3,:)<0));
% PhaseMap                = reshape(Phase, [nr nc]);
% 
% clear C Phase Amp
