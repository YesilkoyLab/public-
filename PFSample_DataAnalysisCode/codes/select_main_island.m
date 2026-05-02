function Zlong = select_main_island(Z, frac)
% SELECT_MAIN_ISLAND
%   Zlong = select_main_island(Z, frac)
%   For each row of Z:
%     - shift so min = 0
%     - keep values >= frac * max (e.g. frac=0.3 -> top 70%)
%     - if multiple islands remain, keep ONLY the longest island
%
%   Returns Zlong with NaNs outside main island.

    [nSeg, nWN] = size(Z);
    Z70   = nan(size(Z));

    % thresholding
    for k = 1:nSeg
        t       = Z(k,:);
        t_shift = t - min(t);
        thr     = frac * max(t_shift);
        z       = t_shift;
        z(z < thr) = NaN;
        Z70(k,:) = z;
    end

    % longest island
    Zlong = Z70;
    for k = 1:nSeg
        row  = Zlong(k,:);
        mask = ~isnan(row);
        if ~any(mask), continue; end

        d     = diff([false mask false]);
        start = find(d == 1);
        stop  = find(d == -1) - 1;
        lens  = stop - start + 1;

        [~, iMax] = max(lens);
        keepMask = false(size(mask));
        keepMask(start(iMax):stop(iMax)) = true;

        row(~keepMask) = NaN;
        Zlong(k,:)     = row;
    end
end
