function [ spectra,indices ] = qualityTestClassifier( spectra,wn,peak,trough,minimum,maximum,indices )
%Take only spectra whose amide band signals are between 0.1 and 2.0
%This ensures that we only signal from tissue region, because only those from tissue region have 
%sufficient signal in the amide band

%find index of peak
[~,idxPeak]=min(abs(wn-peak));
%find index of through
[~,idxTrough]=min(abs(wn-trough));
%bad spectra are those which peak-through values are below minimum or above
%maximum
bad=find(spectra(:,idxPeak)-spectra(:,idxTrough)<minimum|spectra(:,idxPeak)-spectra(:,idxTrough)>maximum);

%get rid of bad spectra
spectra(bad,:)=[];


indices(bad)=[];

    



end

