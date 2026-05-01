function output = vectornorm(input, slow)

%
% syntax: [output] = vectornorm(input, slow);
%
% Version 2.1 (was vecnorm_cytospec.m)
% Performs a vector normalisation on the input.
% See http://www.cytospec.com/preproc.html#PreprocNorm
% or
% http://books.google.com/books?id=85-m480rYNMC&pg=PA156&lpg=PA156&dq=%22ve
% ctor+normalisation%22&source=bl&ots=Fxvs1c4LJc&sig=CX8SPnaxSlB8z2lgtOGMxA
% buNic&hl=en&sa=X&oi=book_result&resnum=3&ct=result#PPA156,M1
% or http://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=00820296
%
%   input  - a matrix of variables (spectra in rows)
%   slow   - any variable here will produce a slow version which might get
%               past 'out of memory' errors. You could use a string 'slow'
%               or a 1. The contents of this variable is not used; it is
%               simply a placemarker to indicate the algorithm should take
%               a different route.
%
%   output - the vector normalised values (spectra in rows)
%
% Calculates the following:
%   1. Squares each variable in 'input'.
%   2. Sums these squares and calculates the square root of the result.
%      This is the 'vector length'.
%   3. Divides each of the original data variables by the vector length.
%   4. Outputs the result to a Matlab variable
%
%
% Alex Henderson, February 2007, updated to v2 October 2008
%                                updated November 2008
%                                updated to v2.1 June 2013
%                                   (corrected sparsity of output)            
%   Copyright (c) 2007 - 2013, Alex Henderson 
%   Contact email: alex.henderson@manchester.ac.uk
%   Licenced under the GNU General Public License (GPL) version 3
%   http://www.gnu.org/copyleft/gpl.html
%   Other licensing options are available, please contact Alex for details
%   If you use this file in your work, please acknowledge the author(s) in
%   your publications. 

% Version 2.1, Alex Henderson June 2013
%   Modified output to remain sparse if the input was sparse, with
%   conversion to a dense matrix otherwise. 

% input = load('lncapspectra.csv'); % for debugging purposes

if (exist('slow', 'var') == 0)
    slow = false;
else
    slow = true;
end

% data has dimension [n,m]= size(input)
[rows,cols]=size(input);                % get the dimensions of the input matrix
output=input;

%output=zeros(size(input));
% if (slow)
%     mean_y_value = mean(input, 2);  % average y value ([n,1])
%     for col = 1:cols
%         output(:,col) = input(:,col) - mean_y_value;   % subtract the mean from the original data ([n,m])
%     end
% else
%     mean_y_value_matrix = mean(input, 2)*ones(1,cols);  % average y value ([n,m])
%     output = input - mean_y_value_matrix;
% end

squares = output.^2;                    % square of each absorbance ([n,m])
sum_of_squares = sum(squares, 2);       % sum of the squares along the rows ([n,1])

divisor = sqrt(sum_of_squares);         % ([n,1])

divisor(divisor==0) = 1;                % avoid divide by zero error

if (slow)
    for col = 1:cols
        output(:,col) = output(:,col)./divisor;   % divide the data by the vector length ([n,m])
    end
else
    % generate a sparse matrix where the diagonal is of the inverse of the divisor
    multiplier = 1./divisor;
    multiplierdiag=spdiags(multiplier,0,length(multiplier),length(multiplier));

    output=multiplierdiag*output; % divide the data by the vector length ([n,m])
end

if(~issparse(input))
    output=full(output);
end
    