function [Z, Zlong, A2_long] = process_one_sample(matFile, wn, fracMain, polyOrder, baseWin)
% PROCESS_ONE_SAMPLE
%   [Z, Zlong, A2_long] = process_one_sample(matFile, wn, fracMain, polyOrder, baseWin)
%
% Inputs:
%   matFile   : path to 1 sample .mat file
%   wn        : wavenumber axis (1 x N)
%   fracMain  : threshold fraction (e.g., 0.3 → keep top 70%)
%   polyOrder : SG polynomial order (e.g., 3)
%   baseWin   : nominal SG window (e.g., 15)
%
% Outputs:
%   Z       : raw averaged 20 x N transmittance
%   Zlong   : thresholded + longest-island per metasurface
%   A2_long : 2nd derivative absorbance (same size as Z/Zlong)

    % ---- 1) Load and spatially average → Z ----
    Z = load_and_average_sample(matFile);

    % ---- 2) Select main island region per metasurface → Zlong ----
    Zlong = select_main_island(Z, fracMain);

    % ---- 3) Compute SG second derivative → A2_long ----
    dt = wn(2) - wn(1);
    A2_long = sg_second_derivative(Zlong, wn, polyOrder, baseWin, dt);

end
