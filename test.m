clear all;
N = 21;
xin = linspace(0,1,21);
y = -sin(.8*pi*xin);

w1 = sdpvar(3,1);
b1 = sdpvar(3,1);
x1 = sdpvar(3,N);
w2 = sdpvar(1,3);
b2 = sdpvar(1,1);


res = w2*x1+b2-y;
f = res*res';

p = @(x) max(0,x).^2;
g1 = (x1-(w1*xin+repmat(b1,1,N)));

ops = sdpsettings('usex0',1);
sigma = 10;
for i = 1:4
    pen = sum(sum(p(-g1))) + ...
          sum(sum(p(-x1))) + ...
          sum(sum(p(g1.*x1)));
    obj = f + sigma*pen;
    optimize([],obj,ops);
    
    x1s = poslin(value(w1)*xin+value(b1));
    ys = value(w2)*x1s+value(b2);

    hold off
    plot(xin,y,'Linewidth',3);
    hold on
    plot(xin,ys,'g--','Linewidth',4);
    value(obj)
    sigma = sigma*10;
end


%con = [f1 >= 0; x1 >= 0; f1.*x1 <= 0;];
%con2 = [x1 == max(w1*xin+repmat(b1,1,N),0);
%        x2 == max(w2*x1 +repmat(b2,1,N),0)];

%%



