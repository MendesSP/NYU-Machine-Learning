%Show EigenFaces

filename = 'Yale_64x64.mat';
data = load(filename); %load data
X = data.fea; 
y = data.gnd;

XMax = max(max(X));
X = X / XMax;
XMean = mean(X); 
m= size(X);

rawData = X - repmat(XMean , m(1) , 1);
covMatrix = rawData * rawData' / (m(2) - 1);

[eigenVectors,eigenValues] = eig(covMatrix);
newData = rawData' * eigenVectors ;

[eigenValues,index]=sort(diag(eigenValues), 'descend');
W = newData (: , index );

for i=1:size(W,2)
    W(:, i) =W(:, i) / norm(W(:,i));
end

WMax = max(max(W)); WMin = min(min(W));
dataW = (W - WMin) ./ (WMax - WMin);

%plot data
% figure; hold on;
% for i =1:4
%     subplot(2,2, i);
%     eigenFace = reshape(dataW(: , i ) , 64, 64);
%     imshow( eigenFace );
% end

%Calculate 80% of energy
i=1;
m= size(X);
rawData = X - repmat(XMean , m(1) ,1);
totalEnergy = sum( diag ( rawData *rawData'));
newEnergy = 0;

while (newEnergy < 0.8 * totalEnergy)
    newData =W(:, 1:i)' * rawData';
    newEnergy = sum(diag(newData' * newData));
    i=i+1;
end

n=i;