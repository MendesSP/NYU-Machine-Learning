function acc = libsvmcrossval_ova(y, X, opts, nfold, indices)
    if nargin < 3, opts = ''; end
    if nargin < 4, nfold = 10; end
    if nargin < 5, indices = crossvalidation(y, nfold); end

    %# N-fold cross-validation testing
    acc = zeros(nfold,1);
    for i=1:nfold
        testIdx = (indices == i); trainIdx = ~testIdx;
        mdl = libsvmtrain_ova(y(trainIdx), X(trainIdx,:), opts);
        [~,acc(i)] = libsvmpredict_ova(y(testIdx), X(testIdx,:), mdl);
    end
    acc = mean(acc);    %# average accuracy
end