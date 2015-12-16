function mdl = libsvmtrain_ova(y, X, opts)
    if nargin < 3, opts = ''; end

    %# classes
    labels = unique(y);
    numLabels = numel(labels);

    %# train one-against-all models
    models = cell(numLabels,1);
    for k=1:numLabels
        models{k} = svmtrain(double(y==labels(k)), X, strcat(opts,' -b 1 -q'));
    end
    mdl = struct('models',{models}, 'labels',labels);
end