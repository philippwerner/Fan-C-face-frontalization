function [D, I] = util_pdist2_fast(varargin)
% D = util_pdist2_fast(X,Y, distance, 'Smallest', K)

    if nargin < 2
        error('You must provide at least two arguments.');
    end

    X = varargin{1};
    Y = varargin{2};
    I = [];

    if size(X, 2) ~= size(Y, 2)
        error('X and Y must have the same number of columns.');
    end

    if nargin == 2
        method = 'euclidean';
    else
        method = varargin{3};
    end

    % Calculate distance
    switch method
        case 'euclidean'
            D = sqrt( bsxfun(@plus,sum(X.^2,2), sum(Y.^2,2)') - 2*(X*Y') );
        otherwise
            error('Not yet implemented.');
    end

    if nargin == 4
        error('Sry, not yet implemented.');
    end

    if nargin == 5
        small_large = varargin{4};
        K = varargin{5};

        switch small_large
            case 'Smallest'
                if K == 1
                    [D,I] = min(D,[],1);
                    return;
                end
                [D,I] = sort(D,1,'ascend');
            case 'Largest'
                if K == 1
                    [D,I] = max(D,[],1);
                    return;
                end
                [D,I] = sort(D,1,'descend');
            otherwise
                error('Wrong input parameter 4: Only ''Smallest'' or ''Largest'' allowed.');
        end

        D = D(1:K,:);
        I = I(1:K,:);
    end
    
    if nargin > 5
        error('Too many input parameters.');
    end

end