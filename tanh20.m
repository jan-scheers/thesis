R = zeros(20,6);
for k = 1:20
    
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
    net.divideParam.testInd = testInd;
    
    
    N = 0.8*N;
    xin = x(trainInd);

    w1 = sdpvar(3,1);
    b1 = sdpvar(3,1);
    x1 = sdpvar(3,N);
    w2 = sdpvar(3,3,'full');
    b2 = sdpvar(3,1);
    x2 = sdpvar(3,N);
    w3 = sdpvar(1,3);
    b3 = sdpvar(1,1);


    assign(w1,net.IW{1});
    assign(b1,net.b{1});
    assign(x1,tansig(value(w1*xin+repmat(b1,1,N))));
    assign(w2,net.LW{2,1});
    assign(b2,net.b{2});
    assign(x2,tansig(value(w2*x1 +repmat(b2,1,N))));
    assign(w3,net.LW{3,2});
    assign(b3,net.b{3});

    res = w3*x2 + b3 - y(trainInd);
    obj = res*res';
    
    net.trainFcn = 'traingd';
    net.trainParam.epochs = 2000;
    net.trainParam.max_fail = 200;
    [net, tr] = train(net,x,y);

    con = [x1 == tansig(w1*xin+repmat(b1,1,N));
           x2 == tansig(w2*x1 +repmat(b2,1,N))];
    ops = sdpsettings('usex0',1,'solver','fmincon');
    ops.fmincon.MaxFunEvals = 20000;
    ops.fmincon.MaxIter = 40;

    t0 = cputime;
    optimize(con,obj,ops);
    t1 = cputime-t0;
    
    x1s = tansig(value(w1)*x(testInd)+value(b1));
    x2s = tansig(value(w2)*x1s+value(b2));
    ysm = value(w3)*x2s+value(b3);
    
    R(k,:) = [tr.perf(end),tr.tperf(end),tr.time(end),value(obj),immse(ysm,y(testInd)),t1]
end
%%
% figure(2);
% hold off
% 
% plot(x,-sin(.8*pi*x),'b-','Linewidth',2)
% hold on;
% 
% xsm = linspace(0,1,1000);
% x1s = tansig(value(w1)*xsm+value(b1));
% x2s = tansig(value(w2)*x1s+value(b2));
% ysm = value(w3)*x2s+value(b3);
% 
% plot(xsm,ysm,'r-','Linewidth',2);
% 
% plot(x(trainInd),y(trainInd),'k+','Linewidth',2,'Markersize',8);
% plot(x(testInd) ,y(testInd) ,'r+','Linewidth',2,'Markersize',8);
% 
% ylim([-1.2,0])
% 
% legend("-sin(.8\pix)","Neural net fit","Training data","Test data")


