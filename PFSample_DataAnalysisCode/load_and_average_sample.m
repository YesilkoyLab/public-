function Z = load_and_average_sample(matFile)
% LOAD_AND_AVERAGE_SAMPLE
%   Z = load_and_average_sample(matFile)
%   Loads one sample file and returns a 20 x N matrix Z where each row is
%   the 11x11 spatial average of spectra_T_01 ... spectra_T_20.

    data = load(matFile);

    nSpectra  = 20;
    specNames = arrayfun(@(k) sprintf('spectra_T_%02d', k), ...
                         1:nSpectra, 'UniformOutput', false);

    % Peek at first cube to get spectral length
    firstCube = data.(specNames{1});       % 11 x 11 x N
    nWN       = size(firstCube, 3);

    Z = zeros(nSpectra, nWN);

    for i = 1:nSpectra
        cube = data.(specNames{i});       % 11 x 11 x N
        spec = squeeze(mean(mean(cube,1),2));  % N x 1
        Z(i,:) = spec(:).';
    end
end
