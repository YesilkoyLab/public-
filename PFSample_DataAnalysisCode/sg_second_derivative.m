function A2_long = sg_second_derivative(Zlong, wn, polyOrder, baseWin, dt)
% SG_SECOND_DERIVATIVE
%   A2_long = sg_second_derivative(Zlong, wn, polyOrder, baseWin, dt)
%   Computes 2nd derivative of absorbance for each row of Zlong using
%   Savitzky–Golay, ignoring NaNs and blanking edges.
%
%   Zlong :  nSeg x nWN, NaNs outside valid region
%   wn    :  1 x nWN wavenumber axis
%   polyOrder : SG polynomial order (e.g. 3)
%   baseWin   : nominal SG window (e.g. 15)
%   dt    : wavenumber step (e.g. 2)

    [nSeg, nWN] = size(Zlong);
    A2_long = nan(size(Zlong));

    for k = 1:nSeg
        Trow = Zlong(k,:);
        mask = ~isnan(Trow);
        if nnz(mask) < 7, continue; end

        Tk   = Trow(mask);
        % wn_k = wn(mask);   % not strictly needed here

        % safe absorbance
        eps_val = 1e-8;
        Tk_safe = max(Tk, eps_val);
        A  = -log10(Tk_safe);

        % SG window per row
        len = numel(A);
        win = min(baseWin, len);
        if mod(win,2) == 0, win = win-1; end
        if win < 5, continue; end

        [~, G] = sgolay(polyOrder, win);
        g2 = factorial(2)/dt^2 * G(:,3);

        A2k = conv(A, g2, 'same');

        % blank unreliable edges
        halfWin    = floor(win/2);
        valid_idx  = (1+halfWin):(len-halfWin);
        A2k_edge   = nan(size(A2k));
        A2k_edge(valid_idx) = A2k(valid_idx);

        row2       = nan(1, nWN);
        row2(mask) = A2k_edge;
        A2_long(k,:) = row2;
    end
end
