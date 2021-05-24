N = 20;
x = linspace(0,1,N);
y = -sin(0.8*pi*x)+normrnd(0,0.1,size(x));
[trainInd,valInd,testInd] = dividerand(N,0.8,0,0.2);

net = fitnet([3 3]);
net = configure(net,x,y);
net.inputs{1}.processFcns = {};
net.outputs{2}.processFcns = {};
net.divideFcn = 'divideind';
net.divideParam.trainInd = trainInd;
net.divideParam.valInd = valInd;
net.divideParam.testInd = valInd;

net.trainFcn = 'traingd';
net.trainParam.epochs = 2000;
net.trainParam.max_fail = 200;
[net, tr] = train(net,x,y);
%%
figure(2);
hold off

plot(x,-sin(.8*pi*x),'b-','Linewidth',2)
hold on

xsm = linspace(0,1,1000);
plot(xsm,sim(net,xsm),'r-','Linewidth',2);

plot(x(trainInd),y(trainInd),'k+','Linewidth',2,'Markersize',8);
plot(x(testInd) ,y(testInd) ,'r+','Linewidth',2,'Markersize',8);

ylim([-1.2,0])

legend("-sin(.8\pix)","Neural net fit","Training data","Test data")